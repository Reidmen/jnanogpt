import functools
import os


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


print(f"XLA flags: {os.environ.get('XLA_FLAGS', '')}")

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
  vocab_size: int = 50304
  num_layers: int = 12
  num_heads: int = 12
  num_embeds: int = 768
  mlp_expansion_factor: int = 4
  dropout_rate: float = 0.1
  use_bias: bool = True
  dtype: jnp.dtype = jnp.bfloat16
  deterministic: bool = False
  use_proj_bias: bool = True
  remat: bool = False


class SelfAttention(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array, mask: jax.Array, deterministic: bool = False):
    batch_size, seq_len, hidden_dim = x.shape
    assert hidden_dim % self.cfg.num_heads == 0
    head_dim = hidden_dim // self.cfg.num_heads
    qkv = nnx.Dense(features=3 * hidden_dim, use_bias=self.cfg.use_proj_bias, dtype=self.cfg.dtype, name="c_attn")(x)
    qkv = qkv.reshape(batch_size, seq_len, 3 * self.cfg.num_heads, head_dim)
    q, k, v = jnp.array_split(qkv, indices_or_sections=3, axis=2)
    attn = jax.nn.dot_product_attention(q, k, v, bias=None)
    x = nnx.Dense(features=seq_len, use_bias=self.cfg.use_proj_bias, dtype=self.cfg.dtype, name="c_proj")
    x = nnx.Dropout(rate=self.cfg.dropout_rate, deterministic=self.cfg.deterministic)(attn)
    return x


class MLP(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array):
    batch_size, seq_len, hidden_dim = x.shape
    x = nnx.Dense(
      features=self.cfg.mlp_expansion_factor * hidden_dim, dtype=self.cfg.dtype, use_bias=self.cfg.use_bias, name="c_fc"
    )(x)
    x = nnx.gelu(x, approximate=True)
    x = nnx.Dense(features=hidden_dim, dtype=self.cfg.dtype, use_bias=self.cfg.use_bias, name="c_proj")(x)
    x = nnx.Dropout(rate=self.cfg.dropout_rate, deterministic=self.cfg.deterministic)(x)
    return x


class Block(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array, mask: jax.Array):
    residual = x
    # Attention block
    x = nnx.LayerNorm(epsilon=self.cfg.epsilon, dtype=self.cfg.dtype, use_bias=self.cfg.use_bias)(x)
    attn_block = SelfAttention
    if "Attn" in self.cfg.remat:
      attn_block = nnx.remat(attn_block, prevent_cse=False)
    x = attn_block(self.cfg, name="attn")(x, mask)
    x = x + residual
    # MLP block
    mlp_block = MLP
    if "MLP" in self.cfg.remal:
      mlp_block = nnx.remat(mlp_block, prevent_cse=False)
    x = mlp_block(self.cfg)(x)
    return x


class GPTModel(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, idx: jax.Array):
    batch_size, seq_len = idx.shape
    position = jnp.arange(0, seq_len)[:, None]
    attn_mask = nnx.make_causal_mask(idx, dtype=bool)

    wte = nnx.Embed(self.cfg.vocab_size, self.cfg.num_embeds, dtype=self.cfg.dtype, name="wte")
    wpe = nnx.Embed(self.cfg.block_size, self.cfg.num_embeds, dtype=self.cfg.dtype, name="wpe")
    token_embed = wte(idx)  # (batch_size, seq_len, num_embed)
    position_embed = wpe(position)  # (1, seq_len, num_embeds)
    x = nnx.Dropout(rate=self.cfg.dropout_rate)(token_embed + position_embed)
    for i in range(self.cfg.num_layers):
      x = Block(self.cfg, name=f"block_{i}")(x, attn_mask)
    x = nnx.LayerNorm(self.cfg.epsilon, dtype=self.cfg.dtype, use_bias=self.cfg.use_bias, name="ln_f")(x)
    logits = wte(x).astype(self.cfg.dtype)
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
  params = convert_hf_params(model_hf, config.num_heads, config.num_embeds)
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
  remat: bool = False


@functools.partial(jax.pmap, axis_name="batch")
def train_step(state: TrainState, tokens: jnp.ndarray, dropout_key: jax.random.PRNGKey) -> tuple[jax.Array, jax.Array]:
  dropout_key = jax.random.fold_in(dropout_key, state.step)

  def _loss(params: flax.core.FrozenDict) -> jnp.ndarray:
    X, Y = tokens[:, :-1], tokens[:, 1:]
    logits = state.apply_fn(params, X, False, rngs={"dropout": dropout_key})
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()
    return loss

  # per-devie loss and grads
  loss, grads = jax.value_and_grad(_loss, has_aux=False)(state.params)
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


def init_train_sate(key, lr, cfg: TrainConfig) -> TrainState:
  model = GPTModel(cfg=cfg)
  params = model.init(key)
  optimizer = optax.chain(
    optax.clip_by_global_norm(cfg.grad_clip),
    optax.adamw(lr, **cfg.betas, weight_decay=cfg.weight_decay),
    optax.apply_every(cfg.gradient_accumulation_steps),
  )
  train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


def get_default_config() -> TrainConfig:
  # use this file to set default values
  path = os.environ.get("GPT_CONFIG", os.path.join("config", "gpt2.yaml"))
  if not os.path.exists(path):
    return TrainConfig()
  print(f"using config file at {path}")
  with open(path, "r") as f:
    raise NotImplementedError
    return from_yaml(TrainConfig, f)


if __name__ == "__main__":
  import wandb

  config = tyro.cli(TrainConfig, default=get_default_config())

  if config.wandb is not None and jax.process_index() == 0:
    wandb.init(**asdict(config.wandb))
    wandb.config.update(asdict(config))

  block_size = config.model.block_size

  # ===== datasets =====
  train_ds = get_dataset(
    config.train_pattern, config.batch_size, block_size, config.shuffle_buffer_size, seed=config.seed
  )

  val_ds = get_dataset(config.val_pattern, config.batch_size, block_size, repeat=1)

  # =====  init parameters ============
  key = jax.random.PRNGKey(config.seed)
  key, key_params, key_dropout = jax.random.split(key, 3)
  # make sure dropout keys are different for each device (local and global)
  key_dropout = jax.random.fold_in(key_dropout, jax.process_index())
  keys_dropout = jax.random.split(key_dropout, jax.local_device_count())

  # ===== learning rate schedule =====
  learning_rate = optax.warmup_cosine_decay_schedule(**asdict(config.learning_rate))

  train_state = init_train_state(key_params, config, learning_rate)

  num_params = count_params(train_state.params)
  if jax.process_index() == 0:
    # logging.info(f'PARAMETER COUNT: {num_params:,}')
    print(f"PARAMETER COUNT: {num_params:,}")

  best_val_loss = float("inf")

  # ==== restore dataset and train state ==== #
  # restore unreplicated optimizer + model state from last checkpoint.
  # this is a no-op if no checkpoints exist
  train_state = checkpoints.restore_checkpoint(f"{config.out_dir}/checkpoints/train_state", train_state)

  # grab step from last checkpoint
  step = int(train_state.step)

  train_iter = iter(train_ds)
  # We need to be able to save the dataset state for stopping and resuming training
  # we'll save a dataset checkpoint for each shard
  dataset_manager = tf.train.CheckpointManager(
    tf.train.Checkpoint(iterator=train_iter),
    f"{config.out_dir}/checkpoints/dataset_{jax.process_index()}",
    max_to_keep=config.keep_checkpoints,
  )
  dataset_manager.restore_or_initialize()

  # replicate parameters to each device
  train_state = replicate(train_state)

  for step in range(step, config.train_steps):
    if step % config.eval_interval == 0:
      val_loss = evaluate(train_state, val_ds, config.batch_size, block_size, config.eval_steps)

      if config.eval_only:
        break

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        if jax.process_index() == 0:
          # save train state in process 0
          checkpoints.save_checkpoint(
            f"{config.out_dir}/checkpoints/train_state",
            unreplicate(train_state),
            step,
            keep=config.keep_checkpoints,
            overwrite=True,
          )
        dataset_manager.save(step)

      if (config.wandb is not None) and (jax.process_index() == 0):
        wandb.log({"val/loss": val_loss}, step=step)

    tokens = next(train_iter)._numpy()
    loss, train_state = train_step(train_state, tokens, keys_dropout)

    if (config.wandb is not None) and (jax.process_index() == 0):
      wandb.log(
        {
          "train/loss": loss[0].item(),
          "lr": learning_rate(step) if callable(learning_rate) else learning_rate,
          "step": step,
          "block": step * config.batch_size * jax.device_count(),
        },
        step=step,
      )


"""
# TESTING

hf_config = GPT2Config(
    vocab_size=256,
    n_positions=32,
    n_embd=64,
    n_layer=1,
    n_head=2,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-6,
    use_cache=False
)

config = GPTConfig(
    vocab_size=256,
    block_size=32,
    num_embeds=64,
    num_layers=1,
    num_heads=2,
    dropout_rate=0.1,
)


def test_gpt2():
    key = jax.random.PRNGKey(0)
    key, key_idxs, key_params = jax.random.split(key, 3)

    hf_model = FlaxGPT2LMHeadModel(hf_config)
    hf_params = hf_model.init_weights(key_params, (2, 32))
    model = GPT(config)

    params = model.init(key_params)
    target_shapes = jax.tree_util.tree_map(lambda a: a.shape, params)
    params = convert_hf_params(hf_params, 2, 64)
    shapes = jax.tree_util.tree_map(lambda a: a.shape, params)

    assert shapes == target_shapes

    for k in ('ln_f', 'wpe', 'wte'):
        assert params['params'][k] == hf_params['transformer'][k]

    idxs = jax.random.randint(key_idxs, (2, 32), 0, 256)
    y1 = hf_model(idxs, params=hf_params).logits
    y2 = model.apply(params, idxs, True)
    assert jnp.allclose(y1, y2, atol=1e-6)
"""
