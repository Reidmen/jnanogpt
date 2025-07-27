from functools import partial
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
import flax.struct as fstruct
import flax.linen as nnx
import numpy
import dataclasses

Pytree = Any
Metrics = dict[str, tuple[jax.Array, ...]]

def create_named_sharding(mesh, args: tuple[str | None,...]):
  return NamedSharding(mesh, PartitionSpec(args))


## Model



# Configs
static_compatible_dataclass = lambda cls: tree_util.register_static(dataclasses.dataclass(cls, frozen=True)) # type: ignore

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
  min_weight_size: int = 2**7
  data_axis_name: str = "data"
  remat: tuple[str, ...] = ("MLP", "Attention")


@static_compatible_dataclass
class CosineDecayConfig:
  init_value: float = 0.0
  peak_value: float = 1.0 # 2.5e-4
  warmup_steps: int = 2 # 2000
  decay_steps: int = 5 # 150000
  end_value: float = 1e-4 # 1e-5

@static_compatible_dataclass
class TrainConfig:
  learning_date: float = 1e-3 #4e-4
  num_minibatches: int = 2 
  grad_clip: float = 1.0 
  gradient_accumulation_steps: int = 1
  cosine_decay_config: CosineDecayConfig = dataclasses.field(default_factory=CosineDecayConfig)


@static_compatible_dataclass
class Config:
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
  seed: int = 32


class TrainState(flax_train_state.TrainState):
  rng: jax.Array


@fstruct.dataclass
class Batch:
  inputs: jax.Array
  labels: jax.Array


def accumulate_gradients_loop(
  state: TrainState, batch: Batch, loss_fn: Callable, num_minbatches: int, rng: jax.Array
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

  return grads, metrics # type: ignore


def accumulated_gradients_scan(
  state: TrainState, batch: Batch, loss_fn: Callable, num_minbatches: int, rng: jax.Array
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
  rng: jax.Array,
  num_minbatches: int,
  loss_fn: Callable,
  use_scan: bool = False,
) -> tuple[Pytree, Metrics]:
  if use_scan:
    return accumulated_gradients_scan(state, batch, loss_fn, num_minbatches, rng)
  else:
    return accumulate_gradients_loop(state, batch, loss_fn, num_minbatches, rng)


# FIXME: remove, using jax.nn.dot_product_attention
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
    batch_size, seq_len, _ = x.shape
    assert self.cfg.hidden_size % self.cfg.head_dim == 0
    num_heads = self.cfg.hidden_size // self.cfg.head_dim
    hidden_size = self.cfg.hidden_size

    dense_module = shard_module_params(
      nnx.Dense,
      axis_name=self.cfg.data_axis_name,
      min_weight_size=self.cfg.min_weight_size
    )
    qkv = dense_module(features=3 * hidden_size, dtype=self.cfg.dtype, name="c_attn")(x)
    qkv = qkv.reshape(batch_size, seq_len, 3 * num_heads, self.cfg.head_dim)
    q, k, v = jnp.array_split(qkv, indices_or_sections=3, axis=2)  # (batch_size, seq_len, num_heads, head_dim)
    attn = jax.nn.dot_product_attention(q, k, v, bias=None)  # add mask
    attn = attn.reshape((batch_size, seq_len, hidden_size))
    x = dense_module(features=self.cfg.embed_size, dtype=self.cfg.dtype, name="c_proj")(attn)
    x = nnx.Dropout(rate=self.cfg.dropout_rate, deterministic=not self.cfg.train)(x)
    return x


class MLP(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array):
    hidden_size = self.cfg.hidden_size
    dense_module = shard_module_params(
      nnx.Dense,
      axis_name=self.cfg.data_axis_name,
      min_weight_size=self.cfg.min_weight_size
    )
    x = dense_module(features=self.cfg.mlp_expansion * hidden_size, dtype=self.cfg.dtype, name="c_fc")(x)
    x = nnx.gelu(x, approximate=True)
    x = dense_module(features=self.cfg.embed_size, dtype=self.cfg.dtype, name="c_proj")(x)
    x = nnx.Dropout(rate=self.cfg.dropout_rate, deterministic=not self.cfg.train)(x)
    return x


class TransformerBlock(nnx.Module):
  cfg: ModelConfig

  @nnx.compact
  def __call__(self, x: jax.Array, mask: jax.Array):
    residual = x  # (batch_size, seq_len, num_embed)
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
    x = mlp_block(cfg=self.cfg)(x)
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

## Sharding -- FSDP
Parameter = jax.Array | nnx.Partitioned # fully replicated or partioned params 

@jax.named_scope("shard_params")
def shard_params(
  params: Pytree, axis_name: str, min_weight_size: int = 2**9
) -> Pytree:
  axis_idx = jax.lax.axis_index(axis_name)
  axis_size = jax.lax.psum(1, axis_name)

  def _split(x: Parameter) -> Parameter:
    if isinstance(x, nnx.Partitioned):
      value, names = x.value, x.names
    else: # is jax.Array
      value = x
      names = (None,) * value.ndim
    if axis_name in names or value.size <= min_weight_size:
      # print(f"Parameter {value.shape} {names} sharded on axis {axis_name} or size {value.size} <= 2**9")
      return x
    else:
      shape = value.shape
      idx = numpy.argsort(shape)[::-1]
      for i in idx:
        if shape[i] % axis_size == 0 and names[i] is None:
          split_size = shape[i] // axis_size
          param_sharded = nnx.Partitioned(
            value=jax.lax.dynamic_slice_in_dim(value, axis_idx * split_size, split_size, axis=i),
            names=names[:i] + (axis_name,) + names[i+1:]
          )
          return param_sharded
      print(f"Could not shard {value.shape} {names} on axis {axis_name} ")
      return x
  
  return jax.tree_util.tree_map(_split, params, is_leaf=lambda x: isinstance(x, nnx.Partitioned))

def gather_array_with_mean_grads(x: jax.Array, axis: int, axis_name: str):
  axis_size = jax.lax.psum(1, axis_name)

  @jax.custom_gradient
  def _inner_fn(x):
    def grad_fn(g): # Computes average grads
      return jax.lax.psum_scatter(g, axis_name, scatter_dimension=axis, tiled=True) / axis_size
    # Gathers x across replicas, returning with the avg. grads
    return jax.lax.all_gather(x, axis_name, axis=axis, tiled=True), grad_fn
  
  return _inner_fn(x)


@jax.named_scope("gather_params")
def gather_params(params: Pytree, axis_name: str) -> Pytree:
  def _gather(p: Parameter) -> Parameter:
    if isinstance(p, nnx.Partitioned) and axis_name in p.names:
      param_shard = p.names
      axis_shard = param_shard.index(axis_name)
      value = gather_array_with_mean_grads(p.value, axis=axis_shard, axis_name=axis_name)
      param_shard = param_shard[:axis_shard] + (None,) + param_shard[axis_shard+1:]
      if any([name is not None for name in param_shard]):
        return nnx.Partitioned(value, param_shard)
      else:
        return value # only when all axis are replicated 
    else:
      return p
  
  return jax.tree_util.tree_map(_gather, params, is_leaf=lambda x: isinstance(x, nnx.Partitioned))


def shard_module_params(
    target: nnx.Module | Callable, axis_name: str, min_weight_size: int = 2 ** 9
) -> nnx.Module | Callable:
  return nnx.map_variables(
    target,
    trans_in_fn=partial(gather_params, axis_name=axis_name),
    trans_out_fn=partial(shard_params, axis_name=axis_name, min_weight_size=min_weight_size),
    mapped_collections="params",
    mutable=True
  )


def synchronize_gradients(grads: Pytree, axis_names: tuple[str,...]) -> Pytree:
  """Synchronize gradients across devices."""
  def _sync_grad(g: Parameter):
    if isinstance(g, nnx.Partitioned):
      replication_axis_names = [
        name for name in axis_names if name not in jax.tree_util.tree_leaves(g.names)
      ]
      if len(replication_axis_names) == 0: # parameters partitioned over all axes 
        return g
      else: # avg over remaining replicated axes
        return g.replace(value=jax.lax.pmean(g.value, axis_name=replication_axis_names))
    else: # parameters are replicated over all axes
      return jax.lax.pmean(g, axis_name=axis_names)
  return jax.tree_util.tree_map(_sync_grad, grads, is_leaf=lambda x: isinstance(x, nnx.Partitioned))

## Train 

def fold_rng_over_axis(rng: jax.Array, axis_name: str) -> jax.Array: # PRNG
  axis_index = jax.lax.axis_index(axis_name)
  return jax.random.fold_in(rng, axis_index)

def loss_fn(params: Pytree, apply_fn: Any, batch: Batch, rng: jax.Array, axis_name: str) -> tuple[jax.Array, dict[str, Any]]:
  dropout_rng = fold_rng_over_axis(rng, axis_name)
  # apply_fn comes from state.apply_fn
  logits = apply_fn({"params": params}, batch.inputs, rngs={"dropout": dropout_rng})
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
  corrected_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)
  batch_size = batch.inputs.shape[0]
  step_metrics = {"loss": (loss.sum(), batch_size), "accuracy": (corrected_pred.sum(), batch_size)}
  return loss.mean(), step_metrics


def train_step(state: TrainState, metrics: Metrics | None, batch: Batch, cfg: Config) -> tuple[TrainState, Metrics]:
  rng, step_rng = jax.random.split(state.rng)
  loss_fn_with_axis = partial(loss_fn, axis_name=cfg.model.data_axis_name)
  grads, step_metrics = accumulate_gradients(
    state, batch, step_rng, cfg.train.num_minibatches, loss_fn_with_axis 
  )
  with jax.named_scope("synchronize_gradients"):
    grads = synchronize_gradients(grads, (cfg.model.data_axis_name,))
  new_state = state.apply_gradients(grads=grads, rng=rng)
  with jax.named_scope("synchronize_metrics"):
    step_metrics = jax.tree_util.tree_map(lambda x: jax.lax.psum(x, axis_name=cfg.model.data_axis_name), step_metrics)
  if metrics is None:
    metrics = step_metrics
  else:
    metrics = jax.tree_util.tree_map(jnp.add, metrics, step_metrics)
  return new_state, metrics # type: ignore

def init_train_state(
  rng: jax.Array, input: jax.Array, model: nnx.Module, optimizer: Any
) -> TrainState:
  init_rng, rng = jax.random.split(rng)
  variables = model.init({"params": init_rng}, input)
  params = variables.pop("params")
  train_state = TrainState.create(rng=rng, apply_fn=model.apply, params=params, tx=optimizer)
  return train_state


def contruct_optimizer(cfg: TrainConfig):
  learning_rate = optax.warmup_cosine_decay_schedule(**dataclasses.asdict(cfg.cosine_decay_config)) # type: ignore
  optimizer = optax.chain(
    optax.clip_by_global_norm(cfg.grad_clip),
    optax.adamw(learning_rate),
    optax.apply_every(cfg.gradient_accumulation_steps),
  ) 
  return optimizer

def test_model():
  key = jax.random.PRNGKey(64)
  rng, input_rng, model_rng = jax.random.split(key, 3)
  cfg = ModelConfig(vocab_size=256, num_layers=4, embed_size=128, train=False) # type: ignore
  train_cfg = TrainConfig()
  gpt_model = GPTModel(cfg=cfg)
  batch_size, seq_len = 8, 32
  init_input = jax.random.randint(input_rng, (batch_size, seq_len), minval=0, maxval=cfg.vocab_size)
  mesh = jax.make_mesh((8,), axis_names=(cfg.data_axis_name,))
  optimizer = contruct_optimizer(train_cfg)
  init_fsdp_fn = jax.shard_map(
    partial(init_train_state, model=gpt_model, optimizer=optimizer),
    in_specs=(PartitionSpec(), PartitionSpec(cfg.data_axis_name)),
    out_specs=PartitionSpec(),
    mesh=mesh,
    check_vma=False
  )
  state_fsdp_shapes = jax.eval_shape(init_fsdp_fn, model_rng, init_input)
  state_fsdp_specs = nnx.get_partition_spec(state_fsdp_shapes)
  print(f"RNG {state_fsdp_specs.rng}")
  print(f"Parameters\n {state_fsdp_specs.params}")
  print(f"Optimizer state\n {state_fsdp_specs.opt_state}")

  init_fsdp_fn = jax.jit(jax.shard_map(
    partial(init_train_state, model=gpt_model, optimizer=optimizer),
    in_specs=(PartitionSpec(), PartitionSpec(cfg.data_axis_name)),
    out_specs=state_fsdp_specs,
    mesh=mesh,
    check_vma=True
  ))
  state_fsdp = init_fsdp_fn(model_rng, init_input)
  print("FSDP Parameter shape per-layer")
  print(jax.tree_util.tree_map(lambda x: x.shape, jax.device_get(state_fsdp.params)))
  print("FSDP parameter size per-layer")
  params_size = jax.tree_util.tree_map(lambda x: x.size, jax.device_get(state_fsdp.params))
  print(params_size)
  print("FSDP Model size")
  print(jax.tree_util.tree_reduce(lambda a, b: a + b, params_size))



def test_train():
  from pprint import pprint
  key = jax.random.PRNGKey(64)
  rng, input_rng, model_rng = jax.random.split(key, 3)
  cfg = Config(
    model=ModelConfig(vocab_size=128, num_layers=4, embed_size=64, train=True), # type: ignore 
    train=TrainConfig() # type: ignore
  )
  gpt_model = GPTModel(cfg=cfg.model)
  batch_size, seq_len = 16, 32 
  init_inputs = jax.random.randint(input_rng, (batch_size, seq_len), minval=0, maxval=cfg.model.vocab_size)
  batch = Batch(
    inputs=init_inputs,# type: ignore
    labels=jnp.pad(init_inputs[:, :-1], ((0, 0), (1, 0)))# type: ignore
  )
  mesh = jax.make_mesh((8,), axis_names=(cfg.model.data_axis_name,))
  optimizer = contruct_optimizer(cfg.train)
  init_fsdp_fn = jax.shard_map(
    partial(init_train_state, model=gpt_model, optimizer=optimizer),
    in_specs=(PartitionSpec(), PartitionSpec(cfg.model.data_axis_name)),
    out_specs=PartitionSpec(),
    mesh=mesh,
    check_vma=False
  )
  state_fsdp_shapes = jax.eval_shape(init_fsdp_fn, model_rng, batch.inputs)
  state_fsdp_specs = nnx.get_partition_spec(state_fsdp_shapes)
  print(f"RNG {state_fsdp_specs.rng}")
  print("Parameters")
  pprint(state_fsdp_specs.params, indent=1)
  # print("Optimizer")
  # pprint(state_fsdp_specs.opt_state[1][0])


  init_fsdp_fn = jax.jit(jax.shard_map(
    partial(init_train_state, model=gpt_model, optimizer=optimizer),
    in_specs=(PartitionSpec(), PartitionSpec(cfg.model.data_axis_name)),
    out_specs=state_fsdp_specs,
    mesh=mesh,
    check_vma=True
  ))
  state_fsdp = init_fsdp_fn(model_rng, batch.inputs)
  print("Parameters shapes")
  pprint(jax.tree_util.tree_map(lambda x: x.shape, jax.device_get(state_fsdp.params)))
  
  params_size = jax.tree_util.tree_map(lambda x: x.size, jax.device_get(state_fsdp.params))
  print(f"FSDP Model size: {jax.tree_util.tree_reduce(lambda a, b: a + b, params_size)}")

  train_step_fsdp_fn = jax.jit(jax.shard_map(
    partial(train_step, cfg=cfg),
    in_specs=(state_fsdp_specs, PartitionSpec(), PartitionSpec(cfg.model.data_axis_name)),
    out_specs=(state_fsdp_specs, PartitionSpec()),
    mesh=mesh,
    check_vma=False
  ),
  donate_argnames=("state", "metrics"),
  static_argnames=("cfg",)
  )
  _, metric_shapes = jax.eval_shape(train_step_fsdp_fn, state_fsdp, None, batch)
  metrics_fsdp = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)
  for _ in range(100):
    state_fsdp, metrics_fsdp = train_step_fsdp_fn(state_fsdp, metrics_fsdp, batch)
    print(f"Iteration {_}")
    pprint(metrics_fsdp, indent=1)
  # deltas = jax.tree_util.tree_map(lambda a, b: jnp.linalg.norm(a - b), old_prams, state-fsdp.params)
  final_metrics_fsdp = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metrics_fsdp)
  state_fsdp, final_metrics_fsdp = train_step_fsdp_fn(state_fsdp, final_metrics_fsdp, batch)
  print("FSDP - Final metrics")
  print(final_metrics_fsdp)
  



def count_params(params: Any) -> int:
  prms = jax.tree_util.tree_map(lambda a: a.size if isinstance(a, jax.Array) else 0, params)
  return jax.tree_util.tree_reduce(lambda a, b: a + b, prms)


if __name__ == "__main__":
  # test_model()
  test_train()
