"""
Huang et al., 2019] Gpipe: Efficient training of giant neural networks using pipeline parallelism. Advances in neural information processing systems
[Narayanan et al., 2021] Efficient large-scale language model training on gpu clusters using megatron-lm
[Lamy-Poirier, 2023] Breadth-First Pipeline Parallelism. Proceedings of Machine Learning and Systems
[McKinney, 2023] A Brief Overview of Parallelism Strategies in Deep Learning
[Huggingface, 2024] Huggingface, 2024. Model Parallelism
[DeepSpeed, 2024] DeepSpeed, 2024. Pipeline Parallelism
"""

import os

import optax

from main import accumulate_gradients, get_num_params


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


# Check injected flags
if os.getenv("USE_GPU", False):
  set_xla_flags_gpu()
else:
  set_xla_flags_cpu(8)

print(f"XLA flags: {os.environ.get('XLA_FLAGS', '')}")

import jax
from jax.sharding import PartitionSpec
import jax.numpy as jnp
import numpy
import flax
import flax.linen as nnx
import dataclasses
from functools import partial
from typing import Any

Pytree = Any
Metrics = dict[str, tuple[jax.Array, ...]]

static_compatible_dataclass = lambda cls: jax.tree_util.register_static(dataclasses.dataclass(cls))


@static_compatible_dataclass
class ModelConfig:
  # data config
  batch_size: int = 16
  seq_len: int = 32

  # model config
  hidden_size: int = 256  # 1024
  dropout_rate: float = 0.1
  mlp_expansion: int = 4
  num_layers: int = 12
  head_dim: int = 64  # 128
  vocab_size: int = 512
  causal_mask: bool = True
  max_seq_len: int = 64
  num_outputs: int = 512  # 2048

  # dtypes & lax optimizations
  dtype: jnp.dtype = jnp.bfloat16
  softmax_dtype: jnp.dtype = jnp.float32
  scan_layers: bool = False
  remat: tuple[str, ...] = ("MLP", "Attention")

  # parallelism
  pipeline_axis_name: str = "model"
  pipeline_axis_size: int = 4
  data_axis_name: str = "data"
  data_axis_size: int = 2


@static_compatible_dataclass
class OptimizerConfig:
  learning_rate: float = 4e-4
  num_minibatches: int = 4


@static_compatible_dataclass
class Config:
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
  seed: int = 32


# high-level modules
class MLPBlock(nnx.Module):
  config: ModelConfig
  train: bool

  @nnx.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    input_features = x.shape[-1]
    residual = x
    x = nnx.LayerNorm(dtype=self.config.dtype, name="pre_norm")(x)
    x = nnx.Dense(
      features=self.config.hidden_size * self.config.mlp_expansion, dtype=self.config.dtype, name="input_dense"
    )(x)
    x = nnx.silu(x)
    x = nnx.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(x)
    return x + residual


class MLPLayers(nnx.Module):
  config: ModelConfig
  train: bool

  @nnx.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    block_module = MLPBlock
    if "MLP" in self.config.remat:
      block_module = nnx.remat(block_module, prevent_cse=False)
    block = block_module(config=self.config, train=self.train, name="block")
    x, _ = nnx.scan(
      lambda module, carry, _: (module(carry), ()),
      variable_axes={"params": 0},
      split_rngs={"params": True, "dropout": True},
      length=self.config.num_layers,
    )(block, x, ())
    # Without scan
    # for i in range(self.config.num_layers):
    #   x = block_module(config=self.config, train=self.train, name=f"block_{i}")(x)


class TrainState(flax.training.train_state.TrainState):
  rng: jax.Array


@flax.struct.dataclass
class Batch:
  inputs: jax.Array
  labels: jax.Array


def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str) -> jax.random.PRNGKey:
  axis_index = jax.lax.axis_index(axis_name)
  return jax.random.fold_in(rng, axis_index)


def sync_gradients(grads: Pytree, axis_names: tuple[str, ...]) -> Pytree:
  def _sync_grad(g: nnx.Partitioned) -> nnx.Partitioned:
    replication_axis_names = [name for name in axis_names if name not in jax.tree_util.tree_leaves(g.names)]
    if len(replication_axis_names) == 0:
      return g
    else:
      return g.replace(value=jax.lax.pmean(g.value, axis_name=replication_axis_names))

  return jax.tree_util.tree_map(_sync_grad, grads, is_leaf=lambda x: isinstance(x, nnx.Partitioned))


# parallelism

Parameter = jax.Array | nnx.Partitioned


def stack_params(params: Pytree, axis_name: str, axis: int = 0, mask_except: int | None = None) -> Pytree:
  def _stack(x: Parameter) -> Parameter:
    if isinstance(x, nnx.Partitioned):
      value, names = x.value, x.names
    else:
      value, names = x, (None,) * x.ndim

    if mask_except is not None:
      axis_index = jax.lax.axis_index(axis_name)
      value = jnp.where(axis_index == mask_except, value, 0.0)

    value = jnp.expand_dims(value, axis)  # for staking on axis
    names = names[:axis] + (axis_name,) + names[axis:]
    return nnx.Partitioned(value, names=names)

  return jax.tree_util.tree_map(_stack, params, is_leaf=lambda x: isinstance(x, nnx.Partitioned))


def unstack_params(params: Pytree, axis_name: str) -> Pytree:
  def _unstack(x: Parameter) -> Parameter:
    if isinstance(x, nnx.Partitioned) and axis_name in x.names:
      value, names = x.value, x.names
      axis_index = names.index(axis_name)
      value = value.squeeze(axis_index)
      names = names[:axis_index] + names[(axis_index + 1) :]
      if all([_name is None for _name in names]):
        return value
      else:
        return nnx.Partitioned(value, names=names)
    else:
      return nnx.Partitioned(value, names=names)

  return jax.tree_util.tree_map(_unstack, params, is_leaf=lambda x: isinstance(x, nnx.Partitioned))


@jax.named_scope("pipeline_step")
def pipeline_step(
  module: nnx.Module, state: jax.Array, input: jax.Array, *args, model_axis_name: str, **kwargs
) -> tuple[jax.Array, jax.Array]:
  num_stages = jax.lax.psum(1, axis_name=model_axis_name)  # find the total stages
  stage_index = jax.lax.axis_index(axis_name=model_axis_name)
  # Initial stage, input is the microbatch
  state = jnp.where(stage_index == 0, input, state)
  state = module(state, *args, **kwargs)
  # Final stage, state is the output
  output = jnp.where(stage_index == num_stages - 1, state, jnp.zeros_like(state))
  # Ring over all other stages
  state = jax.lax.ppermute(state, model_axis_name, perm=[(i, (i + 1) % num_stages) for i in range(num_stages)])
  return (state, output)


@jax.named_scope("execute_pipeline")
def execute_pipeline(
  module: nnx.Module, x: jax.Array, *args, num_microbatches: int, model_axis_name: str, **kwargs
) -> jax.Array:
  """Using GPipe priciple for splitting the batch into microbatches"""
  num_stages = jax.lax.psum(1, axis_name=model_axis_name)
  batch_size = x.shape[0]
  if batch_size % num_microbatches != 0:
    raise Exception(f"{batch_size=} must be divisible by {num_microbatches=}")
  microbatch_size = batch_size // num_microbatches
  microbatches = jnp.reshape(x, (num_microbatches, microbatch_size, *x.shape[1:]))
  input_array = jnp.concatenate(
    [microbatches, jnp.zeros((num_stages - 1, *microbatches.shape[1:]), dtype=microbatches.dtype)], axis=0
  )  # Add zeros for unused computation blocks for first stage

  state = jnp.zeros_like(microbatches[0])
  num_iterations = input_array.shape[0]

  _, outputs = nnx.scan(
    partial(pipeline_step, *args, model_axis_name=model_axis_name, **kwargs),
    variable_broadcast={"params": True},
    split_rngs={"params": False, "dropout": True},
    length=num_iterations,
    in_axes=0,
    out_axes=0,
  )(module, state, input)
  # Take last num_microbatches, all the rest are zeros.
  outputs = jnp.concatenate(outputs[-num_microbatches:], axis=0)
  return outputs


class PipelineModule(nnx.Module):
  axis_name: str
  num_mbatches: int
  module_fn: callable[..., nnx.Module]

  @nnx.compact
  def __call__(self, *args, **kwargs):
    module = self.module_fn()
    return execute_pipeline(module, *args, self.num_mbatches, self.axis_name, **kwargs)


class ModelParallelismModule(nnx.Module):
  axis_name: str
  module_fn: callable[..., nnx.Module]
  module_kwargs: ModelConfig
  mask_except_idx: int | None = None
  split_rngs: bool = True

  @nnx.compact
  def __call__(self, *args, **kwargs):
    if self.is_initializing() and self.split_rngs:
      self.scope.rngs["params"] = self.scope.rngs["params"].replace(
        rng=fold_rng_over_axis(self.scope.rngs["params"].rng, self.axis_name)
      )
    module = nnx.map_variables(
      target=partial(self.module_fn, name="sharded", **self.module_kwargs),
      trans_in_fn=partial(unstack_params, axis_name=self.axis_name),
      trans_out_fn=partial(stack_params, axis_name=self.axis_name, mask_except=self.mask_except_idx),
      mapped_collections="params",
      mutable=True,
    )()
    return module(*args, **kwargs)


class PPClasifier(nnx.Module):
  config: ModelConfig
  pipeline_module: callable[..., nnx.Module] = PipelineModule

  @nnx.compact
  def __call__(self, x: jax.Array, train: bool) -> jax.Array:
    # Input layer -- Only needed for the first stage
    x = ModelParallelismModule(
      module_fn=partial(nnx.Dense, features=self.config.hidden_size, dtype=self.config.dtype),
      axis_name=self.config.model_axis_name,
      mask_except_idx=0,
      name="input_layer",
    )(x)
    # Pipeline
    stage_module_fn = partial(MLPLayers, config=self.config, train=train, name="mlp_layers")
    pipeline_module_fn = partial(
      self.pipeline_module,
      axis_name=self.config.model_axis_name,
      num_mbatches=self.config.num_microbatches,
      module_fn=stage_module_fn,
    )
    module = ModelParallelismModule(
      module_fn=pipeline_module_fn, axis_name=self.config.model_axis_name, name="pipeline"
    )
    x = module(x)
    # Output -- Only needed for the last stage
    parallelism_module = partial(
      ModelParallelismModule, axis_name=self.config.pipeline_axis_name, mask_except_idx=self.config.pipeline_axis_size
    )
    x = parallelism_module(module_fn=partial(nnx.LayerNorm, dtype=self.config.dtype), name="output_norm")(x)
    x = parallelism_module(
      module_fn=partial(nnx.Dense, features=self.config.num_outputs, dtype=self.config.dtype), name="output_layer"
    )(x)
    return x.astype(jnp.float32)


def init_fn(rng: jax.random.PRNGKey, x: jax.Array, model: nnx.Module, optimizer: callable[...]) -> TrainState:
  init_rng, rng = jax.random.split(rng)
  variables = model.init({"params": init_rng}, x, train=False)
  params = variables.pop("params")
  state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer, rng=rng)
  return state


def loss_fn(  # SPMD
  params: Pytree, apply_fn: Any, batch: Batch, rng: jax.Array, cfg: Config
) -> tuple[jax.Array, dict[str, Any]]:
  dropout_rng = fold_rng_over_axis(rng, (cfg.model.data_axis_name, cfg.model.pipeline_axis_name))
  logits = apply_fn({"params": params}, batch.inputs, train=True, rngs={"dropout": dropout_rng})
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
  model_idx = jax.lax.axis_index(cfg.model.pipeline_axis_name)
  model_size = jax.lax.psum(1, cfg.model.pipeline_axis_name)
  # masking loss and prediction for last stage only
  loss = jnp.where(model_idx != model_size - 1, 0.0, loss)
  correct_prediction = jnp.where(model_idx != model_size - 1, False, correct_prediction)
  batch_size = jnp.where(model_idx != model_size - 1, 0, batch.inputs.shape[0])
  # collect metrics
  step_metrics = {"loss": (loss.sum(), batch_size), "accuracy": (correct_prediction.sum(), batch_size)}
  loss = loss.mean()
  return loss, step_metrics


def pp_train_step(state: TrainState, metrics: Metrics | None, batch: Batch, cfg: Config) -> tuple[TrainState, Metrics]:
  rng, step_rng = jax.random.split(state.rng)
  grads, step_metrics = accumulate_gradients(
    state,
    batch,
    step_rng,
    cfg.optimizer.num_minibatches,
    loss_fn=loss_fn,
  )
  with jax.named_scope("sync_gradients"):
    grads = sync_gradients(grads, (cfg.model.data_axis_name, cfg.model.pipeline_axis_name))
  new_state = state.apply_gradients(grads=grads, rng=rng)
  with jax.named_scope("sync_metrics"):
    step_metrics = jax.tree_util.tree_map(
      lambda x: jax.lax.psum(x, axis_name=(cfg.model.data_axis_name, cfg.model.pipeline_axis_name)), step_metrics
    )
  if metrics is None:
    metrics = step_metrics
  else:
    metrics = jax.tree_util.tree_map(jnp.add, metrics, step_metrics)

  return new_state, metrics


if __name__ == "__main__":
  cfg = Config()
  device_array = numpy.array(jax.devices()).reshape(-1, cfg.model.pipeline_axis_name)
  mesh = jax.sharding.Mesh(device_array, (cfg.model.data_axis_name, (cfg.model.pipeline_axis_name)))

  pp_model = PPClasifier(config=cfg)
  optimizer = optax.adamw(learning_rate=cfg.optimizer.learning_rate)

  rng = jax.random.PRNGKey(cfg.seed)
  model_init_rng, data_input_rng, data_labels_rng = jax.random.split(rng, 3)
  batch = Batch(
    input=jax.random.normal(data_input_rng, (cfg.model.batch_size, cfg.model.seq_len)),
    labels=jax.random.randint(data_labels_rng, (cfg.model.batch_size,), 0, cfg.model.vocab_size),
  )
  pp_init_fn = jax.shard_map(
    partial(init_fn, model=pp_model, optimizer=optimizer),
    mesh,
    in_specs=(PartitionSpec(), PartitionSpec(cfg.model.data_axis_name)),
    out_specs=PartitionSpec(),
  )
  pp_state_shapes = jax.eval_shape(pp_init_fn, model_init_rng, batch.inputs)
  pp_state_specs = nnx.get_partition_spec(pp_state_shapes)
  print(pp_state_specs.params)

  # Jitted version
  pp_init_fn = jax.jit(
    jax.shard_map(
      partial(init_fn, model=pp_model, optimizer=optimizer),
      mesh,
      in_specs=(PartitionSpec(), PartitionSpec(cfg.model.data_axis_name)),
      out_specs=pp_state_specs,
    )
  )
  pp_state = pp_init_fn(model_init_rng, batch.inputs)
  print(jax.tree_util.tree_map(lambda x: x.shape, pp_state.params["pipeline"]["sharded"]))

  pp_train_step_fn = jax.jit(
    jax.shard_map(
      pp_train_step,
      mesh,
      in_specs=(pp_state_specs, PartitionSpec(), PartitionSpec(cfg.model.data_axis_name)),
      out_specs=(pp_state_specs, PartitionSpec()),
    ),
    donate_argnames=("state", "metrics"),
  )

  _, metric_shapes = jax.eval_shape(pp_train_step_fn, pp_state, None, batch)
  pp_metrics = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)
  pp_state, pp_metrics = pp_train_step_fn(pp_state, pp_metrics, batch)
  print(f"Number of parameters: {get_num_params(pp_state)}")

  for _ in range(15):
    pp_state, pp_metrics = pp_train_step_fn(pp_state, pp_metrics, batch)

  pp_final_metrics = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)
  print(pp_final_metrics)
