"""Transfomers over a single device"""

import os, subprocess, sys


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

from functools import partial
import jax
import jax.numpy as jnp
from jax import tree_util
import dataclasses
import flax.training
import flax.training.train_state
from typing import Any, Callable
import flax
import flax.linen as nnx
import numpy
import optax

Pytree = Any
Metrics = dict[str, tuple[jax.Array, ...]]


def install_package(package: str) -> None:
  subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])


# Configs
static_compatible_dataclass = lambda cls: tree_util.register_static(dataclasses.dataclass(cls))


@static_compatible_dataclass
class DataConfig:
  batch_size: int = 16  # 64
  seq_len: int = 32  # 512
  vocab_size: int = 512  # 2048


@static_compatible_dataclass
class ModelConfig:
  hidden_size: int = 256  # 1024
  dropout_rate: float = 0.1
  mlp_expansion: int = 4
  num_layers: int = 12
  head_dim: int = 64  # 128
  vocab_size: int = 512
  causal_mask: bool = True
  max_seq_len: int = 64
  num_outputs: int = 512  # 2048
  dtype: jnp.dtype = jnp.bfloat16
  softmax_dtype: jnp.dtype = jnp.float32
  scan_layers: bool = False
  remat: tuple[str, ...] = ("MLP", "Attention")


@static_compatible_dataclass
class OptimizerConfig:
  learning_date: float = 4e-4
  num_minibatches: int = 4


@static_compatible_dataclass
class Config:
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
  data: DataConfig = dataclasses.field(default_factory=DataConfig)
  seed: int = 32


class TrainState(flax.training.train_state.TrainState):
  rng: jax.Array


@flax.struct.dataclass
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
  # Follows https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.dot_product_attention,
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


class MLPBlock(nnx.Module):
  config: ModelConfig
  train: bool

  @nnx.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    hidden_dim = x.shape[-1]
    x = nnx.LayerNorm(dtype=self.config.dtype, name="pre_norm")(x)
    x = nnx.Dense(features=self.config.mlp_expansion * hidden_dim, dtype=self.config.dtype, name="input_layer")(x)
    x = nnx.gelu(x)
    x = nnx.Dense(features=hidden_dim, dtype=self.config.dtype, name="output_layer")(x)
    x = nnx.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(x)
    return x


class AttentionBlock(nnx.Module):
  config: ModelConfig
  mask: jax.Array | None
  train: bool

  @nnx.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    hidden_dim = x.shape[-1]  # x (batch_size, seq_len, head_dim)
    x = nnx.LayerNorm(dtype=self.config.dtype, name="pre_norm")(x)
    num_heads = self.config.hidden_size // self.config.head_dim
    qkv = nnx.DenseGeneral(features=(num_heads, self.config.head_dim * 3), dtype=self.config.dtype, name="qkv")(x)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    x = dot_product_attention(q, k, v, mask=self.mask, dtype=self.config.softmax_dtype)
    x = nnx.DenseGeneral(features=hidden_dim, axis=(-2, -1), dtype=self.config.dtype, name="post_norm")(x)
    x = nnx.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(x)
    return x


class TransformerBlock(nnx.Module):
  config: ModelConfig
  mask: jax.Array | None
  train: bool

  @nnx.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    mlp = MLPBlock
    if "MLP" in self.config.remat:
      mlp = nnx.remat(MLPBlock, prevent_cse=False)
    x = x + mlp(config=self.config, train=self.train, name="mlp")(x)
    attn = AttentionBlock
    if "Attention" in self.config.remat:
      attn = nnx.remat(attn, prevent_cse=False)
    x = x + attn(config=self.config, mask=self.mask, train=self.train, name="attn")(x)
    return x


class Transformer(nnx.Module):
  config: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array, mask: jax.Array | None = None, train: bool = True):
    if mask is not None and self.config.causal_mask:
      mask = nnx.make_causal_mask(x, dtype=jnp.bool_)
    x = nnx.Embed(
      num_embeddings=self.config.vocab_size, features=self.config.hidden_size, dtype=self.config.dtype, name="embed"
    )(x)
    positional_embedding = self.param(
      "positional_embed", nnx.initializers.normal(stddev=0.02), (self.config.max_seq_len, self.config.hidden_size)
    )
    positional_embedding = positional_embedding.astype(self.config.dtype)
    x = x + positional_embedding[None, : x.shape[1]]

    block_fn = partial(TransformerBlock, config=self.config, mask=mask, train=train)
    if "Block" in self.config.remat:
      block_fn = nnx.remat(block_fn, prevent_cse=False)
    if self.config.scan_layers:
      block = block_fn(name="block")
      x, _ = nnx.scan(
        lambda module, carry, _: (module(carry), None),
        variable_axes={"params": 0},
        split_rngs={"params": True, "dropout": True},
        length=self.config.num_layers,
      )(block, x, ())
    else:
      for idx in range(self.config.num_layers):
        x = block_fn(name=f"block_{idx}")(x)

    x = nnx.LayerNorm(dtype=self.config.dtype, name="post_norm")(x)
    x = nnx.Dense(features=self.config.num_outputs, dtype=self.config.dtype, name="output_layer")(x)
    x = x.astype(jnp.float32)
    return x


def next_token_pred_loss(params: Pytree, apply_fn: Any, batch: Batch, rng: jax.Array) -> tuple[Pytree, Metrics]:
  logits = apply_fn({"params": params}, batch.inputs, train=True, rngs={"dropout": rng})
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
  correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)
  batch_size = numpy.prod(batch.labels.shape)
  step_metrics = {"loss": (loss.sum(), batch_size), "accuracy": (correct_pred.sum(), batch_size)}
  loss = loss.mean()
  return loss, step_metrics


@partial(jax.jit, donate_argnames=("state", "metrics"))
def train_step(state: TrainState, batch: Batch, metrics: Metrics | None, cfg: Config) -> tuple[TrainState, Metrics]:
  rng, step_rng = jax.random.split(state.rng)
  grads, step_metrics = accumulate_gradients(
    state,
    batch,
    step_rng,
    cfg.optimizer.num_minibatches,
    loss_fn=next_token_pred_loss,
    use_scan=True,
  )
  new_state = state.apply_gradients(grads=grads, rng=rng)
  if metrics is None:
    metrics = step_metrics
  else:
    metrics = jax.tree_util.tree_map(jnp.add, metrics, step_metrics)
  return new_state, metrics


if __name__ == "__main__":
  cfg = Config()
  model = Transformer(config=cfg.model)
  # Alternative, use Muon optimizer
  # https://github.com/MoonshotAI/Kimi-K
  lr = optax.warmup_exponential_decay_schedule(
    init_value=0.0, peak_value=cfg.optimizer.learning_date, warmup_steps=10, transition_steps=1, decay_rate=0.99
  )
  optimizer = optax.adam(learning_rate=lr)

  tokens = jax.random.randint(
    jax.random.PRNGKey(0), (cfg.data.batch_size, cfg.data.seq_len), minval=1, maxval=cfg.data.vocab_size
  )
  batch = Batch(inputs=jnp.pad(tokens[:, :-1], ((0, 0), (1, 0)), constant_values=0), labels=tokens)

  model_rgn, state_rng = jax.random.split(jax.random.PRNGKey(cfg.seed))
  init_input = batch.inputs[: (cfg.data.batch_size // cfg.optimizer.num_minibatches)]
  model_init = model.init(model_rgn, init_input, mask=None, train=False)
  # print(jax.tree_util.tree_map(jnp.shape, model_init))
  state = TrainState.create(apply_fn=model.apply, params=model_init["params"], tx=optimizer, rng=state_rng)
  print(f"Number of parameters: {get_num_params(state)}")

  _, metric_shapes = jax.eval_shape(train_step, state, batch, None, cfg)
  metrics = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)

  for _ in range(1):
    state, metrics = train_step(state, batch, metrics, cfg)

  final_metrics = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)
  state, final_metrics = train_step(state, batch, final_metrics, cfg)

  jax.profiler.start_trace("traces/")
  for i in range(3):
    with jax.profiler.StepTraceAnnotation("train_step", step_num=i + 1):
      state, metrics = train_step(state, batch, metrics, cfg)

  metrics["loss"][0].block_until_ready()
  jax.profiler.stop_trace()


# to load tensorboard jupyter
# Requires !pip install --upgrade --quiet tensorflow tensorboard_plugin_profile
# %load_ext tensorboard
# %tensorboard --logdir=traces
