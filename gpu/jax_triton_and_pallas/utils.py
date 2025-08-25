#!/usr/bin/env python3
'''Utilties'''

import math
import timeit

import jax
import jax.numpy as jnp

def benchmark(stmt, num_runs=10):
  """Benchmark a function with given arguments."""
  timer = timeit.Timer(stmt)
  return timer.timeit(num_runs) / num_runs

def bw_GBps(size_bytes, time_seconds):
  """Calculate bandwidth in GB/s."""
  return size_bytes / (time_seconds * 1e9)

def tensor_size(tensor_shape: tuple | list, tensor_dtype: jnp.dtype) -> int:
  """Calculate the size of a tensor in bytes."""
  return math.prod(tensor_shape) * jnp.array([], dtype=tensor_dtype).itemsize