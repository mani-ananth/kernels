#!/usr/bin/env python3
'''Softmax using JAX Pallas and JAX Triton.'''

import timeit
import tabulate

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

import triton
import triton.language as tl
import jax_triton as jt

import utils

BLOCK_SIZE = 8  # rows per block
PROBLEM_SHAPE = (1024*16, 1024*16)  # input shape for test 
DTYPE = jnp.float32  # data type for tensors and computation

### REFERENCE IMPLEMENTATIONS
def softmax_naiive(x: jnp.ndarray) -> jnp.ndarray:
  """Compute row wise softmax for a 2D array - Reference 1."""
  assert x.ndim == 2, "Input must be a 2D array"
  max_x = jnp.max(x, axis=-1, keepdims=True)
  e_x = jnp.exp(x - max_x)
  return e_x / jnp.sum(e_x, axis=-1, keepdims=True)

def softmax_jax_nn(x: jnp.ndarray) -> jnp.ndarray:
  """Compute row wise softmax of a 2D array using jax.nn.softmax."""
  assert x.ndim == 2, "Input must be a 2D array"
  return jax.nn.softmax(x)

### PALLAS KERNEL IMPLEMENTATION
def softmax_pallas_kernel_impl(x_ref, o_ref, fake_max: int = 1.09):
  """Pallas kernel implementation of softmax."""
  block_m = x_ref.shape[0]  # rows_per_block
  block_n = x_ref.shape[1]  # n_cols
  row = x_ref[:]
  max_row = jnp.max(row, axis=-1, keepdims=True)
  e_x = jnp.exp(row - max_row)
  o_ref[:] = e_x / jnp.sum(e_x, axis=-1, keepdims=True)

@jax.jit
def softmax_pallas_kernel(x: jax.Array) -> jax.Array:
  assert x.ndim == 2, "Input must be a 2D array"
  n_rows, n_cols = x.shape
  num_blocks = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
  grid_spec = pl.GridSpec(
    grid=(num_blocks, ),
    in_specs=[
      pl.BlockSpec((BLOCK_SIZE, n_cols), lambda i: (i, 0))
    ],
    out_specs=pl.BlockSpec((BLOCK_SIZE, n_cols), lambda i: (i, 0)),
  )
  return pl.pallas_call(
    softmax_pallas_kernel_impl,
    out_shape=jax.ShapeDtypeStruct((n_rows, n_cols), x.dtype),
    grid_spec=grid_spec,
  )(x)

def softmax_pallas(x: jax.Array) -> jax.Array:
  """Compute row wise softmax using Pallas."""
  if x.ndim != 2:
      raise ValueError("Input must be a 2D array")
  grid_dim = (x.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
  return softmax_pallas_kernel(x)

### TRITON KERNEL IMPLEMENTATION
@triton.jit
def softmax_triton_kernel_impl(iptr, optr,
                               irow_stride: tl.constexpr,
                               orow_stride: tl.constexpr,
                               n_rows: tl.constexpr,
                               n_cols: tl.constexpr,
                               num_stages: tl.constexpr):
  row_start = tl.program_id(axis=0)
  row_step = tl.num_programs(axis=0)
  for row_idx in tl.range(row_start, n_rows, row_step):
    # Calc addresses and offsets
    row_start_ptr = iptr + row_idx * irow_stride
    col_offsets = tl.arange(0, n_cols)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    # Load inputs
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    # Compute softmax
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator
    # Store results out
    orow_start_ptr = optr + row_idx * orow_stride
    output_ptrs = orow_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_out, mask=mask)

@jax.jit  
def softmax_triton(x: jax.Array) -> jax.Array:
  assert x.ndim == 2, "Input must be a 2D array"
  n_rows, n_cols = x.shape
  num_blocks = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
  return jt.triton_call(
    x,
    kernel=softmax_triton_kernel_impl,
    out_shape=jax.ShapeDtypeStruct((n_rows, n_cols), x.dtype),
    grid=(num_blocks, ),
    irow_stride=n_rows,
    orow_stride=n_rows,
    n_rows=n_rows,
    n_cols=n_cols,
    num_stages=4,
  )

### TESTING 
def test_softmax(impl1, impl2, shape=(1024, 1024)):
  """Test two softmax implementations for numeric consistency."""
  input_data = jax.random.normal(jax.random.PRNGKey(0), shape, dtype=DTYPE)
  output1 = impl1(input_data)
  output2 = impl2(input_data)
  try:
    assert jnp.allclose(output1, output2, rtol=1e-5, atol=1e-8)
  except AssertionError as e:
    print(f"Softmax implementations differ: {e}")
    print("Input data:\n", input_data)
    print("Output1:\n", output1)
    print("Output2:\n", output2)
    raise

def test_softmax_perf(impl, shape=(1024, 1024), num_runs=10) -> float:
  """Test performance of an implementation."""
  input_data = jax.random.normal(jax.random.PRNGKey(0), shape)
  return utils.benchmark(lambda: impl(input_data).block_until_ready(),
                         num_runs=num_runs)

def main():
  # Test the reference implementations
  test_softmax(softmax_naiive, softmax_jax_nn, PROBLEM_SHAPE)
  test_softmax(softmax_pallas, softmax_jax_nn, PROBLEM_SHAPE)
  test_softmax(softmax_triton, softmax_jax_nn, PROBLEM_SHAPE)
  print("Softmax implementations are consistent!")

  softmax_implementations = {
    "naiive": softmax_naiive,
    "jax_nn": softmax_jax_nn,
    "pallas": softmax_pallas,
    "triton": softmax_triton
  }

  size = utils.tensor_size(PROBLEM_SHAPE, DTYPE) * 2  # input + output

  timing_results = []
  
  for name, func in softmax_implementations.items():
    time_taken = test_softmax_perf(func, PROBLEM_SHAPE)
    GBps = utils.bw_GBps(size, time_taken)
    timing_results.append([name, 10**6 * time_taken, GBps])
  
  print(tabulate.tabulate(
    timing_results,
    headers=["Implementation", "Time (s)", "Bandwidth (GB/s)"],
    tablefmt="grid"
  ))

if __name__ == "__main__":
  main()