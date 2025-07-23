import os
import numpy

def set_xla_flags_gpu():
  flags = os.environ.get("XLA_FLAGS", "")
  flags += (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=false "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
  )
  os.environ["XLA_FLAGS"] = flags


def set_xla_flags_cpu(device_count: int = 8):
  flags = os.environ.get("XLA_FLAGS", "")
  flags += f" --xla_force_host_platform_device_count={device_count}"
  os.environ["XLA_FLAGS"] = flags
  os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Check for temporal flags
if os.getenv("USE_GPU", False):
  set_xla_flags_gpu()
else:
  set_xla_flags_cpu(8)

print(f"XLA flags: {os.environ.get('XLA_FLAGS', '')}")


from typing import Any
import functools
import flax.jax_utils
import flax.training
import flax.training.checkpoints
import flax.core
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import flax
import optax
import tensorflow as tf
from flax.training.train_state import TrainState
import flax.linen as nnx
import dataclasses

## Model

# Configs
static_compatible_dataclass = lambda cls: tree_util.register_static(dataclasses.dataclass(cls))


@static_compatible_dataclass
class ModelConfig:
  block_size: int = 1024
  vocab_size: int = 1024 * 8  # 50304
  num_layers: int = 12
  num_heads: int = 12
  embed_size: int = 768
  epsilon: float = 1e-5
  mlp_expansion_factor: int = 4
  dropout_rate: float = 0.1
  use_bias: bool = True
  dtype: jnp.dtype = jnp.bfloat16
  train: bool = True
  use_proj_bias: bool = True
  remat: tuple[str, ...] = ("Attn", "MLP")
  model_axis_size: int = 8
  model_axis_name: str = "model"


class SelfAttention(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array, mask: jax.Array):
    batch_size, seq_len, hidden_dim = x.shape
    assert hidden_dim % self.cfg.num_heads == 0
    head_dim = hidden_dim // self.cfg.num_heads
    qkv = nnx.Dense(features=3 * hidden_dim, use_bias=self.cfg.use_proj_bias, dtype=self.cfg.dtype, name="c_attn")(x)
    qkv = qkv.reshape(batch_size, seq_len, 3 * self.cfg.num_heads, head_dim)
    q, k, v = jnp.array_split(qkv, indices_or_sections=3, axis=2)  # (batch_size, seq_len, num_heads, head_dim)
    attn = jax.nn.dot_product_attention(q, k, v, bias=None)  # Add mask here
    attn = attn.reshape((batch_size, seq_len, hidden_dim))
    x = nnx.Dense(features=hidden_dim, use_bias=self.cfg.use_proj_bias, dtype=self.cfg.dtype, name="c_proj")(attn)
    x = nnx.Dropout(rate=self.cfg.dropout_rate, deterministic=not self.cfg.train)(x)
    return x


class MLP(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array):
    hidden_dim = x.shape[-1]
    x = nnx.Dense(
      features=self.cfg.mlp_expansion_factor * hidden_dim, dtype=self.cfg.dtype, use_bias=self.cfg.use_bias, name="c_fc"
    )(x)
    x = nnx.gelu(x, approximate=True)
    x = nnx.Dense(features=hidden_dim, dtype=self.cfg.dtype, use_bias=self.cfg.use_bias, name="c_proj")(x)
    x = nnx.Dropout(rate=self.cfg.dropout_rate, deterministic=not self.cfg.train)(x)
    return x


class Block(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array, mask: jax.Array):
    residual = x  # (batch_size, seq_len, hidden_dim)
    # Attention block
    x = nnx.LayerNorm(epsilon=self.cfg.epsilon, dtype=self.cfg.dtype, use_bias=self.cfg.use_bias)(x)
    attn_block = SelfAttention
    if "Attn" in self.cfg.remat:
      attn_block = nnx.remat(attn_block, prevent_cse=False)
    x = attn_block(self.cfg, name="attn")(x, mask)
    x = x + residual
    # MLP block
    mlp_block = MLP
    if "MLP" in self.cfg.remat:
      mlp_block = nnx.remat(mlp_block, prevent_cse=False)
    x = mlp_block(self.cfg)(x)
    return x


class GPTModel(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, idx: jax.Array) -> jax.Array:
    batch_size, seq_len = idx.shape
    position = jnp.arange(0, seq_len)[None, :]
    attn_mask = nnx.make_causal_mask(idx, dtype=bool)

    wte = nnx.Embed(
      num_embeddings=self.cfg.vocab_size, features=self.cfg.embed_size, dtype=self.cfg.dtype, name="wte"
    )  # (vocab_size, embed_dim)
    wpe = nnx.Embed(
      num_embeddings=self.cfg.block_size, features=self.cfg.embed_size, dtype=self.cfg.dtype, name="wpe"
    )  # (vocab_size, embed_dim)
    token_embed = wte(idx)  # (batch_size, seq_len, num_embed)
    position_embed = wpe(position)  # (1, seq_len, num_embed)
    x = nnx.Dropout(rate=self.cfg.dropout_rate, deterministic=not self.cfg.train)(token_embed + position_embed)
    for i in range(self.cfg.num_layers):
      x = Block(self.cfg, name=f"block_{i}")(x, attn_mask)
    x = nnx.LayerNorm(self.cfg.epsilon, dtype=self.cfg.dtype, use_bias=self.cfg.use_bias, name="ln_f")(x)
    logits = wte.attend(x).astype(self.cfg.dtype)
    return logits


def convert_hf_params(model, num_heads: int, num_embeds: int) -> dict:
  params = {}
  for k, v in model.parameters.items():
    params[k] = v
  for k in params.keys():
    if k.endswith("attn.c_attn.kernel"):
      params[k] = params[k].T
    elif k.endwith("attn.c_proj.kernel"):
      params[k] = params[k].T
    elif k.split(".")[1] == "mlp" or k.endswith("kernel"):
      params[k] = params[k].T
    else:
      continue
  return params


def get_pretrained_params(model_type: str) -> tuple[ModelConfig, dict]:
  from transformers import GPT2LMHeadModel

  config: ModelConfig = {
    "gpt2": ModelConfig(num_layers=12, num_heads=12, num_embeds=768),
    "gp2-medium": ModelConfig(num_layers=24, num_heads=16, num_embed=1024),
    "gpt2-large": ModelConfig(num_layers=36, num_heads=20, num_embeds=1280),
    "gpt2-xl": ModelConfig(num_layers=48, num_heads=25, num_embeds=1600),
  }.get(model_type)
  model_hf = GPT2LMHeadModel.from_pretrained(model_type)
  params = convert_hf_params(model_hf, config.num_heads, config.embed_size)
  return config, params


## Train


@dataclasses.dataclass(frozen=True)
class WandbConfig:
  entity: str = "reidmen"
  project: str = "todo"
  name: str = "test"
  mode: str = "disabled"
  notes: str = ""


@dataclasses.dataclass(frozen=True)
class CosineDecayScheduleConfig:
  init_value: float = 0.0
  peak_value: float = 2.5e-4
  warmup_steps: int = 2000
  decay_steps: int = 150000
  end_value: float = 1e-5


@dataclasses.dataclass(frozen=True)
class TrainConfig:
  seed: int = 32
  outdit: str = "./results"
  train_pattern: str = "train_??.tfrecord"
  val_pattern: str = "val_??.tfrecord"
  shuffle_buffer_size: int = 12
  eval_interval: int = 500
  eval_steps: int = 16
  eval_only: bool = False
  keep_checkpoints: int = 3
  batch_size: int = 16
  train_steps: int = 150000
  weight_decay: float = 1e-2
  grad_clip: float = 1.0
  gradient_accumulation_steps: int = 1
  betas: tuple[float, float] = (0.9, 0.95)  # adamw betas
  learning_rate: CosineDecayScheduleConfig = dataclasses.field(default_factory=CosineDecayScheduleConfig)
  wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  # remat: bool = False


@functools.partial(jax.pmap, axis_name="batch")
def train_step(state: TrainState, tokens: jax.Array, dropout_key: jax.random.PRNGKey) -> tuple[jax.Array, jax.Array]:
  dropout_key = jax.random.fold_in(dropout_key, state.step)

  def _loss(params: flax.core.FrozenDict) -> jax.Array:
    X, Y = tokens[:, :-1], tokens[:, 1:]
    logits = state.apply_fn(params, X, rngs={"dropout": dropout_key})
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()
    return loss

  loss, grads = jax.value_and_grad(_loss, has_aux=False)(state.params)  # per-device
  grads = jax.lax.pmean(grads, axis_name="batch")
  loss = jax.lax.pmean(loss, axis_name="batch")
  new_state = state.apply_gradients(grads=grads)
  return loss, new_state


@functools.partial(jax.pmap, axis_name="batch")
def eval_step(state: TrainState, tokens: jnp.ndarray) -> jnp.ndarray:
  X, Y = tokens[:, :-1], tokens[:, 1:]
  logits = state.apply_fn(state.params, X, True)
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
  loss = jax.lax.pmean(loss, axis_name="batch")
  return loss


def evaluate(state: TrainState, ds: tf.data.Dataset, batch_size: int, block_size: int, steps: int) -> jnp.ndarray:
  losses = []
  for _, tokens in zip(range(steps), ds):
    tokens = tokens._numpy()
    loss = eval_step(state, tokens)
    losses.append(loss)
  return jnp.mean(jnp.stack(losses))


def init_train_state(
  model: nnx.Module, key: jax.random.PRNGKey, input: jax.Array, lr: Any, cfg: TrainConfig
) -> TrainState:
  params = model.init(key, input)
  optimizer = optax.chain(
    optax.clip_by_global_norm(cfg.grad_clip),
    optax.adamw(lr, *cfg.betas, weight_decay=cfg.weight_decay),
    optax.apply_every(cfg.gradient_accumulation_steps),
  )
  train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
  return train_state


def get_default_config() -> TrainConfig:
  # use this file to set default values
  path = os.environ.get("GPT_CONFIG", os.path.join("config", "gpt2.yaml"))
  if not os.path.exists(path):
    return TrainConfig()
  print(f"using config file at {path}")
  with open(path, "r") as f:
    raise NotImplementedError
    return from_yaml(TrainConfig, f)


import tensorflow as tf

OPTIONS = tf.data.Options()
OPTIONS.deterministic = True
OPTIONS.autotune.enabled = True


def get_dataset(
  pattern: str,
  batch_size: int = 8,
  block_size: int = 1024,
  shuffle_buffer_size: int | None = None,
  repeat: int | None = None,
  seed: int | None = None,
) -> tf.data.Dataset:
  tf.random.set_seed(seed)
  file_ds = tf.data.Dataset.list_files(pattern, shuffle=bool(shuffle_buffer_size))
  file_ds = file_ds.shard(jax.process_count(), jax.process_index())
  ds = tf.data.TFRecordDataset(file_ds, num_parallel_reads=tf.data.AUTOTUNE)
  feature_description = {
    "ids": tf.io.FixedLenFeature([], tf.string, default_value=""),
  }

  def _parse_proto(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    return tf.io.decode_raw(example["ids"], tf.uint16)

  ds = ds.map(_parse_proto, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.repeat(repeat)

  if shuffle_buffer_size is not None:
    ds = ds.shuffle(shuffle_buffer_size)

  ds = ds.unbatch().batch(block_size + 1, drop_remainder=True)
  if shuffle_buffer_size is not None:
    ds = ds.shuffle(shuffle_buffer_size)

  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.batch(jax.local_device_count(), drop_remainder=True)
  ds = ds.with_options(OPTIONS)
  return ds.prefetch(2)


def test_model():
  key = jax.random.PRNGKey(64)
  rng, input_rng, dropout_rng = jax.random.split(key, 3)
  cfg = ModelConfig(block_size=64, vocab_size=256, num_layers=4, num_heads=4, embed_size=32, train=False)
  model = GPTModel(cfg=cfg)
  batch_size, seq_len = 8, 32
  init_input = jax.random.randint(input_rng, (batch_size, seq_len), minval=0, maxval=cfg.vocab_size)
  init_rngs = {"params": rng, "dropout": dropout_rng}
  params = model.init(init_rngs, init_input)
  print(f"Model size: {count_params(params)}")
  sizes = jax.tree_util.tree_map(lambda x: x.shape, params, is_leaf=lambda x: isinstance(x, jax.Array))
  print(f"Model param sizes: \n{sizes}")
  ret = model.apply(params, init_input)  # (batch_size, seq_len, vocab_size)
  assert ret.shape == (batch_size, seq_len, cfg.vocab_size)


def test_train():
  cfg = ModelConfig(block_size=64, vocab_size=256, num_layers=4, num_heads=4, embed_size=32, train=True)
  init_rng = jax.random.PRNGKey(32)
  rng, input_rng = jax.random.split(init_rng, 2)
  model = GPTModel(cfg=cfg)
  batch_size, seq_len = 8, 32
  init_tokens = jax.random.randint(input_rng, (batch_size, seq_len), minval=0, maxval=cfg.vocab_size)
  # init_rngs = {"params": rng, "dropout": dropout_rng}
  _, key_params, key_dropout = jax.random.split(rng, 3)
  key_dropout = jax.random.fold_in(key_dropout, jax.process_index())
  keys_dropout = jax.random.split(key_dropout, jax.local_device_count())
  train_cfg = TrainConfig()
  learning_rate = optax.warmup_cosine_decay_schedule(**dataclasses.asdict(train_cfg.learning_rate))
  train_state = init_train_state(model, key_params, init_tokens, learning_rate, train_cfg)
  loss, train_state = train_step(train_state, init_tokens, keys_dropout)


def count_params(params: Any) -> int:
  prms = jax.tree_util.tree_map(lambda a: a.size if isinstance(a, jax.Array) else 0, params)
  return jax.tree_util.tree_reduce(lambda a, b: a + b, prms)


def train():
  import wandb

  # Allow user provided and yaml config
  cfg = get_default_config()
  if cfg.wandb is not None and jax.process_index() == 0:
    wandb.init(**cfg.wandb)
    wandb.config.update(**cfg)

  block_size = cfg.model.block_size

  # ===== datasets =====
  train_ds = get_dataset(cfg.train_pattern, cfg.batch_size, block_size, cfg.shuffle_buffer_size, seed=cfg.seed)

  val_ds = get_dataset(cfg.val_pattern, cfg.batch_size, block_size, repeat=1)

  # =====  init parameters ============
  key = jax.random.PRNGKey(cfg.seed)
  key, key_params, key_dropout = jax.random.split(key, 3)
  # make sure dropout keys are different for each device (local and global)
  key_dropout = jax.random.fold_in(key_dropout, jax.process_index())
  keys_dropout = jax.random.split(key_dropout, jax.local_device_count())

  # ===== learning rate schedule =====
  learning_rate = optax.warmup_cosine_decay_schedule(**cfg.learning_rate)

  train_state = init_train_state(key_params, cfg, learning_rate)

  num_params = count_params(train_state.params)
  if jax.process_index() == 0:
    # logging.info(f'PARAMETER COUNT: {num_params:,}')
    print(f"PARAMETER COUNT: {num_params:,}")

  best_val_loss = float("inf")

  # ==== restore dataset and train state ==== #
  # restore unreplicated optimizer + model state from last checkpoint.
  # this is a no-op if no checkpoints exist
  train_state = flax.training.checkpoints.restore_checkpoint(f"{cfg.out_dir}/checkpoints/train_state", train_state)

  # grab step from last checkpoint
  step = int(train_state.step)

  train_iter = iter(train_ds)
  # We need to be able to save the dataset state for stopping and resuming training
  # we'll save a dataset checkpoint for each shard
  dataset_manager = tf.train.CheckpointManager(
    tf.train.Checkpoint(iterator=train_iter),
    f"{cfg.out_dir}/checkpoints/dataset_{jax.process_index()}",
    max_to_keep=cfg.keep_checkpoints,
  )
  dataset_manager.restore_or_initialize()

  # replicate parameters to each device
  train_state = flax.jax_utils.replicate(train_state)

  for step in range(step, cfg.train_steps):
    if step % cfg.eval_interval == 0:
      val_loss = evaluate(train_state, val_ds, cfg.batch_size, block_size, cfg.eval_steps)

      if cfg.eval_only:
        break

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        if jax.process_index() == 0:
          # save train state in process 0
          flax.training.checkpoints.save_checkpoint(
            f"{cfg.out_dir}/checkpoints/train_state",
            flax.jax_utils.unreplicate(train_state),
            step,
            keep=cfg.keep_checkpoints,
            overwrite=True,
          )
        dataset_manager.save(step)

      if (cfg.wandb is not None) and (jax.process_index() == 0):
        wandb.log({"val/loss": val_loss}, step=step)

    tokens = next(train_iter)._numpy()
    loss, train_state = train_step(train_state, tokens, keys_dropout)

    if (cfg.wandb is not None) and (jax.process_index() == 0):
      wandb.log(
        {
          "train/loss": loss[0].item(),
          "lr": learning_rate(step) if callable(learning_rate) else learning_rate,
          "step": step,
          "block": step * cfg.batch_size * jax.device_count(),
        },
        step=step,
      )


if __name__ == "__main__":
  # test_model()
  test_train()
