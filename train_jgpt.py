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
  set_xla_flags_cpu()

print(f"XLA flags: {os.environ.get('XLA_FLAGS', '')}")


from typing import Any, Callable
import flax.jax_utils
import flax.training
import flax.training.checkpoints
import flax.core
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from jax.sharding import PartitionSpec, NamedSharding
import flax
import optax
import tensorflow as tf
from flax.training import train_state as flax_train_state
import flax.linen as nnx
import dataclasses

Pytree = Any
Metrics = dict[str, tuple[jax.Array, ...]]

def create_named_sharding(mesh, args: tuple[str | None,...]):
  return NamedSharding(mesh, PartitionSpec(args))


## Model



# Configs
static_compatible_dataclass = lambda cls: tree_util.register_static(dataclasses.dataclass(cls))

@static_compatible_dataclass
class ModelConfig:
  embed_size: int = 128 # 758
  hidden_size: int = 256  # 1024
  dropout_rate: float = 0.1
  mlp_expansion: int = 4
  num_layers: int = 12
  head_dim: int = 32 # 128
  vocab_size: int = 512
  causal_mask: bool = True
  batch_size: int = 16
  seq_len: int = 32
  max_seq_len: int = 64
  num_outputs: int = 512  # 2048
  epsilon: float = 1e-6
  dtype: jnp.dtype = jnp.bfloat16
  softmax_dtype: jnp.dtype = jnp.float32
  scan_layers: bool = False
  train: bool = False
  remat: tuple[str, ...] = ("MLP", "Attention")


@static_compatible_dataclass
class OptimizerConfig:
  learning_date: float = 4e-4
  num_minibatches: int = 4


@static_compatible_dataclass
class Config:
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
  seed: int = 32


class TrainState(flax_train_state.TrainState):
  rng: jax.Array


class Batch:
  inputs: jax.Array
  labels: jax.Array


def accumulate_gradients_loop(
  state: TrainState, batch: Batch, loss_fn: Callable, num_minbatches: int, rng: jax.random.PRNGKey
) -> tuple[Pytree, Metrics]:
  batch_size = batch.inputs.shape[0]
  minibatch_size = batch_size // num_minbatches
  rngs = jax.random.split(rng, num_minbatches)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  grads, metrics = None, None
  for idx in range(num_minbatches):
    with jax.named_scope(f"minibatch_{idx}"):
      start, end = idx * minibatch_size, (idx + 1) * minibatch_size
      minibatch = jax.tree_util.tree_map(lambda x: x[start:end], batch)
      (_, step_metrics), step_grads = grad_fn(state.params, state.apply_fn, minibatch, rngs[idx])
      if grads is None:
        grads = step_grads
        metrics = step_metrics
      else:
        grads = jax.tree_util.tree_map(jnp.add, grads, step_grads)  # grads += step_grads
        metrics = jax.tree_util.tree_map(jnp.add, metrics, step_metrics)
  grads = jax.tree_util.tree_map(lambda g: g / num_minbatches, grads)
  if metrics is None:
    raise ValueError

  return grads, metrics


def accumulated_gradients_scan(
  state: TrainState, batch: Batch, loss_fn: Callable, num_minbatches: int, rng: jax.random.PRNGKey
) -> tuple[Pytree, Metrics]:
  batch_size = batch.inputs.shape[0]
  minibatch_size = batch_size // num_minbatches
  rngs = jax.random.split(rng, num_minbatches)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  def _minibatch_step(idx: jax.Array | int) -> tuple[Pytree, Metrics]:
    minibatch = jax.tree_util.tree_map(
      lambda x: jax.lax.dynamic_slice_in_dim(x, start_index=idx * minibatch_size, slice_size=minibatch_size), batch
    )
    (_, step_metrics), step_grads = grad_fn(state.params, state.apply_fn, minibatch, rngs[idx])
    return step_grads, step_metrics

  def _scan_step(carry: tuple[Pytree, Metrics], idx: jax.Array | int) -> tuple[tuple[Pytree, Metrics], None]:
    step_grads, step_metrics = _minibatch_step(idx)
    carry = jax.tree_util.tree_map(jnp.add, carry, (step_grads, step_metrics))
    return carry, None

  grads_shape, metrics_shape = jax.eval_shape(_minibatch_step, 0)  # abstract shapes
  grads = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shape)
  metrics = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)
  (grads, metrics), _ = jax.lax.scan(
    _scan_step, init=(grads, metrics), xs=jnp.arange(num_minbatches), length=num_minbatches
  )
  grads = jax.tree_util.tree_map(lambda g: g / num_minbatches, grads)
  return grads, metrics


def accumulate_gradients(
  state: TrainState,
  batch: Batch,
  rng: jax.random.PRNGKey,
  num_minbatches: int,
  loss_fn: Callable,
  use_scan: bool = False,
) -> tuple[Pytree, Metrics]:
  if use_scan:
    return accumulated_gradients_scan(state, batch, loss_fn, num_minbatches, rng)
  else:
    return accumulate_gradients_loop(state, batch, loss_fn, num_minbatches, rng)


def get_num_params(state: TrainState) -> int:
  return sum(numpy.prod(x.shape) for x in jax.tree_util.tree_leaves(state.params))


def dot_product_attention(
  q: jax.Array,  # (..., qseq_len, num_heads, hidden_dim)
  k: jax.Array,  # (..., kseq_len, num_heads, hidden_dim)
  v: jax.Array,  # (..., kseq_len, num_heads, hidden_dim)
  mask: jax.Array,
  dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
  """Naive Attention"""
  hidden_dim = q.shape[-1]
  scale = hidden_dim ** (-0.5)
  q = q * scale
  q, k = q.astype(dtype), k.astype(dtype)  # upcast to "dtype" for stability
  logits = jnp.einsum("...qhd,...khd->...hqk", q, k)  # (..., num_heads, qseq_len, kseq_len)
  if mask is not None:  # causal mask
    logits = jnp.where(mask, logits, jnp.finfo(dtype).min)
  logits = jax.nn.softmax(logits, axis=-1)
  logits = logits.astype(v.dtype)  # downcast to v.dtype
  # (..., qseq_len, num_heads, hidden_dim)
  attn_ret = jnp.einsum("...hqk,...khd->...qhd", logits, v)
  # attn_ret = attn_ret.astype(v.dtype)
  return attn_ret

class AttentionBlock(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array, mask: jax.Array):
    batch_size, seq_len, hidden_dim = x.shape
    assert hidden_dim % self.cfg.head_dim == 0
    num_heads = hidden_dim // self.cfg.head_dim
    qkv = nnx.Dense(features=3 * hidden_dim, dtype=self.cfg.dtype, name="c_attn")(x)
    qkv = qkv.reshape(batch_size, seq_len, 3 * num_heads, self.cfg.head_dim)
    q, k, v = jnp.array_split(qkv, indices_or_sections=3, axis=2)  # (batch_size, seq_len, num_heads, head_dim)
    attn = jax.nn.dot_product_attention(q, k, v, bias=None)  # Add mask
    attn = attn.reshape((batch_size, seq_len, hidden_dim))
    x = nnx.Dense(features=hidden_dim, dtype=self.cfg.dtype, name="c_proj")(attn)
    x = nnx.Dropout(rate=self.cfg.dropout_rate, deterministic=not self.cfg.train)(x)
    return x


class MLP(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array):
    hidden_dim = x.shape[-1]
    x = nnx.Dense(
      features=self.cfg.mlp_expansion * hidden_dim, dtype=self.cfg.dtype, name="c_fc")(x)
    x = nnx.gelu(x, approximate=True)
    x = nnx.Dense(features=hidden_dim, dtype=self.cfg.dtype, name="c_proj")(x)
    x = nnx.Dropout(rate=self.cfg.dropout_rate, deterministic=not self.cfg.train)(x)
    return x


class TransformerBlock(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array, mask: jax.Array):
    residual = x  # (batch_size, seq_len, hidden_dim)
    # Attention block
    x = nnx.LayerNorm(epsilon=self.cfg.epsilon, dtype=self.cfg.dtype)(x)
    attn_block = AttentionBlock 
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
    _, seq_len = idx.shape
    position = jnp.arange(0, seq_len)[None, :] # (1, seq_len)
    attn_mask = nnx.make_causal_mask(idx, dtype=bool)
    pos_embed = nnx.Embed(
      num_embeddings=self.cfg.vocab_size, features=self.cfg.embed_size, dtype=self.cfg.dtype, name="pos_embed"
    )(position)  # (vocab_size, embed_dim) (1, seq_len) -> (1, seq_len, num_embed)
    wte = nnx.Embed(
      num_embeddings=self.cfg.vocab_size, features=self.cfg.embed_size, dtype=self.cfg.dtype, name="tie_embed"
    )  # (vocab_size, embed_dim)
    tok_embed = wte(idx)  # (batch_size, seq_len, num_embed)
    x = nnx.Dropout(rate=self.cfg.dropout_rate, deterministic=not self.cfg.train)(tok_embed + pos_embed)
    # TODO: nnx.scan
    for i in range(self.cfg.num_layers):
      x = TransformerBlock(self.cfg, name=f"block_{i}")(x, attn_mask)
    x = nnx.LayerNorm(
      self.cfg.epsilon, dtype=self.cfg.dtype, name="ln_f"
    )(x)
    logits = wte.attend(x).astype(self.cfg.dtype)
    return logits

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


def train_step(state: TrainState, tokens: jax.Array, dropout_key: jax.random.PRNGKey) -> tuple[jax.Array, TrainState]:
  # dropout_key = jax.random.fold_in(dropout_key, state.step)

  def _loss(params: flax.core.FrozenDict) -> jax.Array:
    X, Y = tokens[:, :-1], tokens[:, 1:]
    logits = state.apply_fn(params, X, rngs={'dropout': dropout_key})
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()
    return loss

  loss, grads = jax.value_and_grad(_loss, has_aux=False)(state.params)  # per-device
  # grads = jax.lax.pmean(grads, axis_name="batch")
  # loss = jax.lax.pmean(loss, axis_name="batch")
  new_state = state.apply_gradients(grads=grads)
  return loss, new_state


def eval_step(state: TrainState, tokens: jnp.ndarray) ->jax.Array: 
  X, Y = tokens[:, :-1], tokens[:, 1:]
  logits = state.apply_fn(state.params, X, True)
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
  # loss = jax.lax.pmean(loss, axis_name="batch")
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
  train_state = TrainState.create(rng=key, apply_fn=model.apply, params=params, tx=optimizer)
  return train_state



def test_model():
  key = jax.random.PRNGKey(64)
  rng, input_rng, dropout_rng = jax.random.split(key, 3)
  cfg = ModelConfig(vocab_size=256, num_layers=4, embed_size=128, train=False) # type: ignore
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
  cfg = ModelConfig(vocab_size=256, num_layers=4, embed_size=32, train=True) # type: ignore
  init_rng = jax.random.PRNGKey(32)
  rng, input_rng = jax.random.split(init_rng, 2)
  model = GPTModel(cfg=cfg)
  batch_size, seq_len = 8, 32
  init_tokens = jax.random.randint(input_rng, (batch_size, seq_len), minval=0, maxval=cfg.vocab_size)
  _, key_params, key_dropout = jax.random.split(rng, 3)
  train_cfg = TrainConfig()
  learning_rate = optax.warmup_cosine_decay_schedule(**dataclasses.asdict(train_cfg.learning_rate))
  train_state = init_train_state(model, key_params, init_tokens, learning_rate, train_cfg)
  loss, train_state = jax.jit(train_step)(train_state, init_tokens, key_dropout)
  assert train_state.step == 1
  assert loss.shape == ()



def count_params(params: Any) -> int:
  prms = jax.tree_util.tree_map(lambda a: a.size if isinstance(a, jax.Array) else 0, params)
  return jax.tree_util.tree_reduce(lambda a, b: a + b, prms)


if __name__ == "__main__":
  test_model()
  test_train()
