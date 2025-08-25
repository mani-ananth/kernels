#!/usr/bin/env python3
'''Vector addition using JAX Pallas and JAX Triton.'''
import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
import jax_triton as jt
import triton
import triton.language as tl

BLOCK_SIZE = 2048
PROBLEM_SIZE = 1024 * 1024

### REFERENCE IMPLEMENTATION ###
def add_vectors_reference(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  if x.shape != y.shape:
      raise ValueError("Input vectors must have the same shape")
  return x + y

### PALLAS KERNEL IMPLEMENTATION ###
def add_vectors_pallas_kernel_impl(x_ref, y_ref, o_ref):
  o_ref[...] = x_ref[...] + y_ref[...]

@jax.jit
def add_vectors_pallas_kernel(x: jax.Array, y: jax.Array, grid: Tuple[int]) -> jax.Array:
  num_blocks = (x.size + BLOCK_SIZE - 1) // BLOCK_SIZE
  grid_spec = pl.GridSpec(
    grid=(num_blocks,),
    in_specs=[
      pl.BlockSpec((BLOCK_SIZE, ), lambda i: (i, )),
      pl.BlockSpec((BLOCK_SIZE, ), lambda i: (i, )),
    ],
    out_specs=pl.BlockSpec((BLOCK_SIZE, ), lambda i: (i, )),
  )
  return pl.pallas_call(
      add_vectors_pallas_kernel_impl,
      out_shape=jax.ShapeDtypeStruct((x.size,), x.dtype),
      grid_spec=grid_spec,
  )(x, y)

def add_vectors_pallas(x: jax.Array, y: jax.Array) -> jax.Array:
  if x.shape != y.shape:
      raise ValueError("Input vectors must have the same shape")
  grid_dim = (x.size + BLOCK_SIZE - 1) // BLOCK_SIZE
  return add_vectors_pallas_kernel(x, y, (grid_dim, ))

### TRITON KERNEL IMPLEMENTATION ###
@triton.jit
def add_vectors_triton_kernel(x_ptr, y_ptr, o_ptr,
                              n_elements: tl.constexpr,
                              BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  start = pid * BLOCK_SIZE
  offsets = start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(o_ptr + offsets, output, mask=mask)

@jax.jit
def add_vectors_triton(x: jax.Array, y: jax.Array) -> jax.Array:
  if x.shape != y.shape:
      raise ValueError("Input vectors must have the same shape")
  
  n_elements = x.size
  num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
  out_shapes = jax.ShapeDtypeStruct(x.shape, x.dtype)
  return jt.triton_call(
    x, y,
    kernel=add_vectors_triton_kernel,
    out_shape=out_shapes,
    grid=(num_blocks, ),
    n_elements=n_elements,
    BLOCK_SIZE=BLOCK_SIZE,
  )

### TEST and COMPARE the two implementations ###
def test_add_vectors(impl1, impl2):
  a = jnp.arange(PROBLEM_SIZE, dtype=jnp.bfloat16)
  b = jnp.arange(PROBLEM_SIZE, dtype=jnp.bfloat16)

  result1 = impl1(a, b)
  result2 = impl2(a, b)

  np.testing.assert_array_equal(result1, result2)

if __name__ == "__main__":
  test_add_vectors(add_vectors_pallas, add_vectors_reference)
  test_add_vectors(add_vectors_triton, add_vectors_reference)
  
  print("All tests passed!")