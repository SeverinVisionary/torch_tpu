# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tolerance overrides for TorchTPU ops tests."""

import collections.abc
import copy

import torch
from torch_tpu._internal.utils import test_utils


def _expm1_atol(y: float) -> float:
  """Returns the absolute tolerance for expm1() given the expected value y."""

  # For expm1 (i.e. y = e^x - 1), we especially care about the precision when
  # y is close to 0. However, a tolerance tight enough for this case is
  # too tight for y >> 0. Without variable-tolerance, we'd be forced to
  # pick a large tolerance for all cases, which would cause unnecessary
  # slack in our tests.
  #
  # Using variable-tolerance, we:
  #   - use a small tolerance for y < 0,
  #   - use a large (relative) tolerance for y > 1,
  #   - linearly interpolate the tolerance for y in [0, 1].
  #
  # This allows the test to be tight for y near 0 or being negative, while
  # allowing more slack for y >> 0.
  atol0 = 3e-6  # small tolerance
  atol1 = 1e-5  # large tolerance
  if y < 0:
    return atol0
  if y > 1:
    return atol1 * y
  return atol0 + (atol1 - atol0) * y


def update_dict(d, u):
  """Recursively update a dictionary to override tolerance values."""
  for k, v in u.items():
    if isinstance(v, collections.abc.Mapping):
      d[k] = update_dict(d.get(k, {}), v)
    else:
      d[k] = v
  return d


# Overrides the default relative and absolute tolerance for the given op and
# dtype between TPU and GPU. Two tensors are considered to match if
#
#   abs(tpu - gpu) <= atol + rtol * abs(gpu)
#
# (A common misconception is that the criteria is
#   abs(tpu - gpu) <= min(atol, rtol * abs(gpu))
# but that's incorrect.)
#
# The overrides apply to both the op and its inplace variant.
#
# The default tolerances are defined here:
#   https://docs.pytorch.org/docs/stable/testing.html
ACCURACY_OVERRIDES_VS_GPU = {
    # go/keep-sorted start
    "_foreach_acos": {
        torch.complex64: {"rtol": 5.2e-5, "atol": 6.7e-5},
    },
    "_foreach_addcdiv": {
        torch.bfloat16: {"rtol": 1.7e-2, "atol": 2.2e-2},
        torch.float16: {"rtol": 1.2e-3, "atol": 3.3e-3},
    },
    "_foreach_addcmul": {
        torch.bfloat16: {"rtol": 2.4e-2, "atol": 2.9e-2},
        torch.float16: {"rtol": 3.1e-3, "atol": 3.8e-3},
    },
    "_foreach_cos": {
        torch.complex64: {"rtol": 5.5e-6},
    },
    "_foreach_cosh": {
        torch.complex64: {"rtol": 5.1e-6},
        torch.float32: {"rtol": 5e-6},
    },
    "_foreach_exp": {
        torch.complex64: {"rtol": 4.8e-6, "atol": 2.8e-6},
        torch.int16: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int32: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int64: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int8: {"rtol": 4.4e-6, "atol": 7.9e-7},
    },
    "_foreach_expm1": {
        torch.complex64: {"rtol": 6.1e-6},
    },
    "_foreach_log": {
        torch.complex64: {"rtol": 6.3e-5, "atol": 6.8e-5},
        torch.float32: {"rtol": 7.4e-5, "atol": 8.3e-5},
    },
    "_foreach_log10": {
        torch.complex64: {"rtol": 3.9e-5, "atol": 5.3e-5},
        torch.float32: {"rtol": 3.3e-5, "atol": 4.2e-5},
    },
    "_foreach_log2": {
        torch.complex64: {"rtol": 1.1e-4, "atol": 3.4e-5},
        torch.float32: {"rtol": 1.1e-4, "atol": 8.1e-5},
    },
    "_foreach_pow": {
        torch.complex64: {"rtol": 8.8e-4, "atol": 3.4e-4},
        torch.float32: {"rtol": 5.5e-6},
    },
    "_foreach_sigmoid": {
        torch.complex64: {"rtol": 1.4e-5},
        torch.float32: {"atol": 2.7e-5},
    },
    "_foreach_tan": {
        torch.complex64: {"rtol": 3.7e-5},
    },
    "_foreach_tanh": {
        torch.complex64: {"rtol": 1.4e-4, "atol": 1.1e-5},
        torch.float32: {"atol": 5.2e-5},
    },
    "_log_softmax_backward_data": {
        torch.float32: {"rtol": 3.5e-5, "atol": 3.6e-5},
    },
    "_native_batch_norm_legit": {
        torch.float32: {"rtol": 1e-5, "atol": 2.1e-5},
    },
    "_native_multi_head_attention": {
        torch.float16: {"rtol": 1.4e-1, "atol": 1.3e-1},
        torch.float32: {"rtol": 2.0e-1, "atol": 3.2e-1},
    },
    "_softmax_backward_data": {
        torch.bfloat16: {"atol": 3.6e-3},
        torch.float16: {"atol": 5.9e-4},
    },
    "_transformer_encoder_layer_fwd": {
        torch.bfloat16: {"rtol": 4.5e-2, "atol": 1.1e-1},
        torch.float16: {"rtol": 7.6e-2, "atol": 1.8e-1},
        torch.float32: {"rtol": 7.0e-2, "atol": 2.1e-1},
    },
    "_upsample_bicubic2d_aa": {
        torch.bfloat16: {"rtol": 3.5e-2, "atol": 3.3e-2},
        torch.float16: {"rtol": 3.8e-3, "atol": 4e-3},
        torch.float32: {"rtol": 2.0e-5, "atol": 2.0e-5},
    },
    "_upsample_bilinear2d_aa": {
        torch.bfloat16: {"rtol": 4.7e-2, "atol": 3.7e-2},
        torch.float16: {"rtol": 6.9e-3, "atol": 6.5e-3},
        torch.float32: {"rtol": 1.7e-5, "atol": 2.5e-5},
    },
    "abs": {
        torch.complex64: {"rtol": 2.0e-6, "atol": 2.0e-6},
        torch.float64: {"rtol": 4.7e-6, "atol": 1e-6},
    },
    "acos": {
        torch.complex64: {"rtol": 5.5e-5, "atol": 2.2e-5},
    },
    "acosh": {
        torch.complex64: {"rtol": 2.9e-5},
        torch.float32: {"rtol": 3e-5, "atol": 1.3e-5},
        torch.int16: {"rtol": 2.8e-5},
        torch.int32: {"rtol": 2.8e-5},
        torch.int64: {"rtol": 2.8e-5},
        torch.int8: {"rtol": 2.8e-5},
        torch.uint8: {"rtol": 2.8e-5},
    },
    "add": {
        torch.bfloat16: {"rtol": 1.8e-2, "atol": 1.5e-2},
        torch.float16: {"rtol": 4.5e-3, "atol": 1.2e-3},
    },
    "addmm": {
        torch.float16: {"rtol": 1.5e-2},
        torch.float32: {"rtol": 7.7e-3},
    },
    "asin": {
        torch.complex64: {"rtol": 7.9e-5, "atol": 1e-4},
    },
    "asinh": {
        torch.bool: {"atol": 2.9e-5},
        torch.complex64: {"rtol": 4.4e-5, "atol": 7.1e-5},
        torch.float32: {"rtol": 7.1e-5, "atol": 8.3e-5},
        torch.int16: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int32: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int64: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int8: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.uint8: {"rtol": 1.6e-5, "atol": 2.9e-5},
    },
    "atan": {
        torch.complex64: {"rtol": 2.1e-5, "atol": 1.8e-5},
    },
    "atanh": {
        torch.complex64: {"rtol": 1.8e-5, "atol": 3.2e-5},
        torch.float32: {"rtol": 5.8e-5, "atol": 7.7e-5},
    },
    "baddbmm": {
        torch.bfloat16: {"rtol": 1.7e-2, "atol": 1.5e-2},
        torch.complex64: {"rtol": 8.3e-2},
        torch.float16: {"rtol": 2.8e-1, "atol": 3.3e-1},
        torch.float32: {"rtol": 3.4e-1, "atol": 3.6e-1},
    },
    "bmm": {
        torch.complex64: {"rtol": 9.6e-2},
        torch.float16: {"rtol": 1.1e-1, "atol": 3.5e-1},
        torch.float32: {"rtol": 6.6e-2, "atol": 3.4e-1},
    },
    "cdist": {
        torch.bfloat16: {"atol": 2.1e-1},
        torch.float16: {"rtol": 2e-1, "atol": 3e-1},
        torch.float32: {"rtol": 5.1e-1, "atol": 3e-1},
    },
    "cos": {
        torch.complex64: {"rtol": 4.5e-6},
    },
    "cosh": {
        torch.complex64: {"rtol": 3.9e-6},
        torch.float32: {"rtol": 3.4e-6},
        torch.int16: {"rtol": 4e-6},
        torch.int32: {"rtol": 4e-6},
        torch.int64: {"rtol": 4e-6},
        torch.int8: {"rtol": 4e-6},
        torch.uint8: {"rtol": 4e-6},
    },
    # bf16/f16 cumsum accumulation rounds differently from the GPU
    # reference, which is itself not bit-exact (mirrors the vs-CPU tol).
    "cumsum": {
        torch.bfloat16: {"atol": 1.9e-2},
        torch.float16: {"rtol": 3.5e-3, "atol": 2.4e-3},
    },
    "digamma": {
        torch.float32: {"rtol": 4e-5, "atol": 1.3e-5},
        torch.int16: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int32: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int64: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.int8: {"rtol": 4.7e-5, "atol": 4.3e-7},
        torch.uint8: {"rtol": 4.7e-5, "atol": 3.6e-5},
    },
    "einsum": {
        torch.bfloat16: {"rtol": 3.9e-2},
        torch.complex64: {"rtol": 2.7e-2},
        torch.float16: {"rtol": 1.4e-2, "atol": 1.5e-1},
        torch.float32: {"rtol": 1.1e-2, "atol": 1.6e-1},
    },
    "erfinv": {
        torch.float32: {"rtol": 1.6e-5, "atol": 1.7e-5},
    },
    "exp": {
        torch.complex64: {"rtol": 2.8e-6},
        torch.float32: {"rtol": 4.6e-6},
        torch.int16: {"rtol": 3.1e-6, "atol": 7.9e-7},
        torch.int32: {"rtol": 3.1e-6, "atol": 7.9e-7},
        torch.int64: {"rtol": 3.1e-6, "atol": 7.9e-7},
        torch.int8: {"rtol": 3.1e-6, "atol": 7.9e-7},
        torch.uint8: {"rtol": 4.4e-6},
    },
    "exp2": {
        torch.bfloat16: {"rtol": 2.4e-2, "atol": 2.4e-3},
        torch.complex64: {"rtol": 3.3e-6},
        torch.float16: {"rtol": 2.3e-3, "atol": 5.9e-4},
        torch.float32: {"rtol": 2.4e-6},
    },
    "expm1": {
        torch.complex64: {"rtol": 4.7e-6},
        torch.float32: {"rtol": 6.9e-6},
        torch.int16: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int32: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int64: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int8: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.uint8: {"rtol": 4.4e-6},
    },
    "floor_divide": {
        torch.bfloat16: {"rtol": 2.1e-1},
    },
    "ldexp": {
        torch.complex64: {"rtol": 4.4e-6},
        torch.float32: {"rtol": 4.1e-6},
    },
    "lgamma": {
        torch.bfloat16: {"rtol": 2.6e-2, "atol": 1.6e-4},
        torch.float16: {"rtol": 1.3e-2, "atol": 1.5e-4},
        torch.float32: {"rtol": 1.3e-2, "atol": 2.6e-4},
        torch.int16: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.int32: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.int64: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.int8: {"rtol": 1.3e-4, "atol": 2.3e-4},
        torch.uint8: {"rtol": 1.3e-4, "atol": 2.3e-4},
    },
    "linalg.lu": {
        torch.float64: {"atol": 2.5e-6},
    },
    "linalg.lu_factor_ex": {
        torch.float32: {"rtol": 2.5e-6, "atol": 2.4e-5},
    },
    "linalg.qr": {
        torch.float32: {"rtol": 1e-4, "atol": 1e-4},
    },
    "linalg.vector_norm": {
        torch.complex64: {"rtol": 4.8e-6},
        torch.float32: {"rtol": 4.8e-6},
    },
    "linspace": {
        torch.bfloat16: {"rtol": 9.2e-3, "atol": 9.4e-3},
        torch.float16: {"rtol": 1.2e-3, "atol": 1.8e-3},
    },
    "log": {
        torch.complex64: {"rtol": 4.7e-5},
        torch.float32: {"rtol": 4.9e-5, "atol": 4.3e-5},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    "log10": {
        torch.complex64: {"rtol": 1.4e-5, "atol": 2.2e-5},
        torch.float32: {"atol": 3.3e-5},
        torch.int16: {"atol": 3.3e-5},
        torch.int32: {"atol": 3.3e-5},
        torch.int64: {"atol": 3.3e-5},
        torch.int8: {"atol": 3.3e-5},
        torch.uint8: {"atol": 3.3e-5},
    },
    "log1p": {
        torch.complex64: {"rtol": 3.3e-5, "atol": 3.7e-5},
        torch.float32: {"rtol": 1.1e-4, "atol": 1.2e-4},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    "log2": {
        torch.complex64: {"rtol": 4.5e-5},
        torch.float32: {"rtol": 5.6e-5, "atol": 2.8e-5},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    "log_softmax": {
        torch.float32: {"rtol": 5.7e-5, "atol": 8.3e-5},
    },
    "logcumsumexp": {
        torch.bfloat16: {"atol": 3.7e-5},
        torch.float16: {"rtol": 1.2e-3, "atol": 5.9e-4},
        torch.float32: {"rtol": 5e-5, "atol": 3.2e-5},
    },
    "logit": {
        torch.bool: {"atol": 7.5e-5},
        torch.float16: {"atol": 5.9e-4},
        torch.float32: {"rtol": 7.5e-5, "atol": 8.3e-5},
        torch.int16: {"atol": 7.5e-5},
        torch.int32: {"atol": 7.5e-5},
        torch.int64: {"atol": 7.5e-5},
        torch.int8: {"atol": 7.5e-5},
        torch.uint8: {"atol": 7.5e-5},
    },
    "matmul": {
        torch.bfloat16: {"rtol": 1.2e-1, "atol": 1.9e-1},
        torch.complex64: {"rtol": 1.4e-1},
        torch.float16: {"rtol": 3e-1, "atol": 4.7e-1},
        torch.float32: {"rtol": 2.7e-1, "atol": 3.9e-1},
    },
    "mm": {
        torch.complex64: {"rtol": 3.8e-2},
        torch.float16: {"rtol": 4.8e-2},
        torch.float32: {"rtol": 5.2e-2},
    },
    # Tolerance overrides justified for performance: native bfloat16/float16
    # pooling computations avoid costly intermediate upcasts to float32.
    "nn.functional.adaptive_avg_pool2d": {
        torch.bfloat16: {"rtol": 1.8e-2, "atol": 2.5e-2},
        torch.float16: {"rtol": 2.5e-3, "atol": 4.5e-3},
    },
    "nn.functional.adaptive_avg_pool3d": {
        torch.bfloat16: {"rtol": 1.8e-2, "atol": 2.0e-2},
        torch.float16: {"rtol": 3.5e-3, "atol": 3.0e-3},
    },
    "nn.functional.avg_pool2d": {
        torch.bfloat16: {"rtol": 3.1e-2, "atol": 4.5e-2},
        torch.float16: {"rtol": 3.1e-3, "atol": 4.1e-3},
    },
    "nn.functional.binary_cross_entropy": {
        torch.bfloat16: {"rtol": 9e-3, "atol": 5.9e-3},
        torch.float16: {"rtol": 8.5e-3, "atol": 4.4e-4},
        torch.float32: {"rtol": 1.8e-4, "atol": 6.9e-5},
    },
    "nn.functional.conv1d": {
        torch.float16: {"rtol": 8.2e-2},
        torch.float32: {"rtol": 9e-2},
    },
    "nn.functional.conv2d": {
        torch.bfloat16: {"atol": 1.9e-2},
        torch.float16: {"rtol": 3.9e-1, "atol": 4.2e-1},
        torch.float32: {"rtol": 4.2e-1, "atol": 4.6e-1},
    },
    "nn.functional.conv_transpose1d": {
        torch.float16: {"rtol": 6.7e-2, "atol": 1.6e-1},
        torch.float32: {"rtol": 7.1e-2, "atol": 2.1e-1},
    },
    "nn.functional.conv_transpose2d": {
        torch.bfloat16: {"rtol": 7.4e-2, "atol": 7.6e-2},
        torch.float16: {"rtol": 2.2e-1, "atol": 6.6e-2},
        torch.float32: {"rtol": 3e-1, "atol": 9.4e-2},
    },
    "nn.functional.embedding": {
        torch.float16: {
            "rtol": 3.9e-3,
            "atol": 2.4e-3,
        },  # TOLERANCE_OVERRIDE_OK
    },
    "nn.functional.embedding_bag": {
        torch.bfloat16: {"atol": 3.2e-2},
        torch.float16: {"atol": 4e-3},
    },
    "nn.functional.group_norm": {
        torch.bfloat16: {"rtol": 3.3e-2, "atol": 8e-2},
    },
    "nn.functional.linear": {
        torch.bfloat16: {"rtol": 8.7e-3, "atol": 1.9e-2},
        torch.complex64: {"rtol": 2.2e-2, "atol": 3.5e-2},
        torch.float16: {"rtol": 1.9e-2, "atol": 1.5e-2},
        torch.float32: {"rtol": 2.0e-2, "atol": 1.7e-2},
    },
    "nn.functional.scaled_dot_product_attention": {
        torch.bfloat16: {"rtol": 3.3e-1, "atol": 4e-1},
        torch.float16: {"rtol": 5e-1, "atol": 5.4e-1},
        torch.float32: {"rtol": 5.6e-1, "atol": 5.6e-1},
    },
    "norm": {
        torch.complex64: {"rtol": 5.7e-6},
        torch.float32: {"rtol": 5.7e-6},
    },
    "polygamma": {
        torch.float32: {"rtol": 1.1e-5},
    },
    "pow": {
        torch.complex64: {"rtol": 7e-4, "atol": 3.8e-4},
        torch.float32: {"rtol": 5.6e-6},
    },
    "sin": {
        torch.complex64: {"rtol": 4.9e-6},
    },
    "sinh": {
        torch.complex64: {"rtol": 4.6e-6},
        torch.float32: {"rtol": 5.8e-6},
        torch.int16: {"rtol": 4e-6},
        torch.int32: {"rtol": 4e-6},
        torch.int64: {"rtol": 4e-6},
        torch.int8: {"rtol": 4e-6},
        torch.uint8: {"rtol": 4e-6},
    },
    "tan": {
        torch.complex64: {"rtol": 3.0e-5, "atol": 5.0e-6},
    },
    "tanh": {
        torch.complex64: {"rtol": 1.1e-5, "atol": 1.1e-5},
        torch.float32: {"atol": 3.8e-5},
        torch.int16: {"atol": 3.6e-5},
        torch.int32: {"atol": 3.6e-5},
        torch.int64: {"atol": 3.6e-5},
        torch.int8: {"atol": 3.6e-5},
        torch.uint8: {"atol": 3.6e-5},
    },
    "torch._scaled_grouped_mm": {
        torch.float8_e4m3fn: {"rtol": 9.4e-3, "atol": 4.7e-3},
    },
    "torch._scaled_mm": {
        torch.float8_e4m3fn: {"rtol": 9.4e-3, "atol": 4.7e-3},
    },
    "torch.ops.aten._safe_softmax.default": {
        torch.float16: {"atol": 5.9e-4},
    },
    "xlogy": {
        torch.float32: {"rtol": 3.2e-4, "atol": 3e-4},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    # go/keep-sorted end
}  # end of ACCURACY_OVERRIDES_VS_GPU

# Like ACCURACY_OVERRIDES_VS_GPU, but with torch.compile().
ACCURACY_OVERRIDES_VS_GPU_COMPILED = {
    # go/keep-sorted start
    "_foreach_cosh": {
        torch.complex64: {"rtol": 5.1e-6},
    },
    "_foreach_exp": {
        torch.complex64: {"rtol": 4.8e-6},
    },
    "_foreach_expm1": {
        torch.complex64: {"rtol": 6.1e-6},
    },
    "_foreach_log": {
        torch.complex64: {"rtol": 6.3e-5, "atol": 6.6e-5},
    },
    "_foreach_log10": {
        torch.complex64: {"rtol": 3.9e-5, "atol": 4.5e-5},
    },
    "_foreach_log2": {
        torch.complex64: {"rtol": 1.1e-4},
    },
    "_foreach_pow": {
        torch.complex64: {"rtol": 8.8e-4, "atol": 3.4e-4},
    },
    "_foreach_sigmoid": {
        torch.complex64: {"rtol": 1.4e-5},
    },
    "_foreach_tan": {
        torch.complex64: {"rtol": 3.7e-5},
    },
    "_foreach_tanh": {
        torch.complex64: {"rtol": 1.4e-4, "atol": 1.1e-5},
    },
    "_log_softmax_backward_data": {
        torch.float32: {"rtol": 3.5e-5, "atol": 3.6e-5},
    },
    "_native_batch_norm_legit": {
        torch.float32: {"rtol": 1e-5, "atol": 2.1e-5},
    },
    "_native_multi_head_attention": {
        torch.float16: {"rtol": 1.4e-1, "atol": 1.3e-1},
        torch.float32: {"rtol": 2.0e-1, "atol": 3.2e-1},
    },
    "_softmax_backward_data": {
        torch.bfloat16: {"atol": 3.6e-3},
        torch.float16: {"atol": 5.9e-4},
    },
    "_upsample_bicubic2d_aa": {
        torch.bfloat16: {"rtol": 3.5e-2, "atol": 3.3e-2},
        torch.float16: {"rtol": 3.8e-3, "atol": 4e-3},
        torch.float32: {"rtol": 2.0e-5, "atol": 2.0e-5},
    },
    "_upsample_bilinear2d_aa": {
        torch.bfloat16: {"rtol": 4.7e-2, "atol": 3.7e-2},
        torch.float16: {"rtol": 6.9e-3, "atol": 6.5e-3},
        torch.float32: {"rtol": 1.7e-5, "atol": 2.5e-5},
    },
    "abs": {
        torch.complex64: {"rtol": 2.0e-6, "atol": 2.0e-6},
        torch.float64: {"rtol": 4.7e-6, "atol": 1e-6},
    },
    "acos": {
        torch.complex64: {"rtol": 5.5e-5, "atol": 2.2e-5},
    },
    "acosh": {
        torch.complex64: {"rtol": 2.9e-5},
    },
    "addcmul": {
        torch.float16: {
            "rtol": 8.6e-3,  # TOLERANCE_OVERRIDE_OK
            "atol": 1.0e-3,  # TOLERANCE_OVERRIDE_OK
        },
    },
    "asin": {
        torch.complex64: {"rtol": 7.9e-5, "atol": 1e-4},
    },
    "asinh": {
        torch.bool: {"atol": 2.9e-5},
        torch.complex64: {"rtol": 4.4e-5, "atol": 7.1e-5},
        torch.float32: {"rtol": 7.1e-5, "atol": 8.3e-5},
        torch.int16: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int32: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int64: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.int8: {"rtol": 1.6e-5, "atol": 2.9e-5},
        torch.uint8: {"rtol": 1.6e-5, "atol": 2.9e-5},
    },
    "atan": {
        torch.complex64: {"rtol": 2.1e-5, "atol": 1.8e-5},
    },
    "atanh": {
        torch.complex64: {"rtol": 1.8e-5, "atol": 3.2e-5},
        torch.float32: {"rtol": 5.8e-5, "atol": 7.7e-5},
    },
    "baddbmm": {
        torch.complex64: {"rtol": 6.9e-2},
    },
    "bmm": {
        torch.complex64: {"rtol": 9.6e-2},
        torch.float16: {"rtol": 1.1e-1, "atol": 3.5e-1},
        torch.float32: {"rtol": 6.6e-2, "atol": 3.4e-1},
    },
    "cos": {
        torch.complex64: {"rtol": 4.5e-6},
    },
    "cosh": {
        torch.complex64: {"rtol": 3.9e-6},
    },
    # bf16/f16 cumsum accumulation rounds differently from the GPU
    # reference, which is itself not bit-exact (mirrors the vs-CPU tol).
    "cumsum": {
        torch.bfloat16: {"atol": 1.9e-2},
        torch.float16: {"rtol": 3.5e-3, "atol": 2.4e-3},
    },
    "einsum": {
        torch.complex64: {"rtol": 2.7e-2},
        torch.float16: {"rtol": 1.4e-2, "atol": 1.5e-1},
        torch.float32: {"rtol": 1.1e-2, "atol": 1.6e-1},
    },
    "erfinv": {
        torch.float32: {"rtol": 1.6e-5, "atol": 1.7e-5},
    },
    "exp": {
        torch.complex64: {"rtol": 2.8e-6},
    },
    "exp2": {
        torch.complex64: {"rtol": 3.3e-6},
    },
    "expm1": {
        torch.complex64: {"rtol": 4.7e-6},
        torch.float32: {"rtol": 6.9e-6},
        torch.int16: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int32: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int64: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.int8: {"rtol": 4.4e-6, "atol": 7.9e-7},
        torch.uint8: {"rtol": 4.4e-6},
    },
    "ldexp": {
        torch.complex64: {"rtol": 4.4e-6},
    },
    "linalg.vector_norm": {
        torch.complex64: {"rtol": 4.8e-6},
        torch.float32: {"rtol": 4.8e-6},
    },
    "linspace": {
        torch.bfloat16: {"rtol": 7.9e-3, "atol": 9.4e-3},
        torch.float16: {"rtol": 1.2e-3, "atol": 1.2e-3},
    },
    "log": {
        torch.complex64: {"rtol": 4.7e-5},
    },
    "log10": {
        torch.complex64: {"rtol": 1.4e-5, "atol": 2.2e-5},
    },
    "log1p": {
        torch.complex64: {"rtol": 3.3e-5, "atol": 3.7e-5},
        torch.float32: {"rtol": 1.1e-4, "atol": 1.2e-4},
        torch.int16: {"rtol": 6.9e-5},
        torch.int32: {"rtol": 6.9e-5},
        torch.int64: {"rtol": 6.9e-5},
        torch.int8: {"rtol": 6.9e-5},
        torch.uint8: {"rtol": 6.9e-5},
    },
    "log2": {
        torch.complex64: {"rtol": 4.5e-5},
    },
    "logcumsumexp": {
        torch.bfloat16: {"atol": 3.7e-5},
        torch.float16: {"rtol": 1.2e-3, "atol": 5.9e-4},
        torch.float32: {"rtol": 5e-5, "atol": 3.2e-5},
    },
    "logit": {
        torch.bool: {"atol": 7.5e-5},
        torch.float16: {"atol": 5.9e-4},
        torch.float32: {"rtol": 7.5e-5, "atol": 8.3e-5},
        torch.int16: {"atol": 7.5e-5},
        torch.int32: {"atol": 7.5e-5},
        torch.int64: {"atol": 7.5e-5},
        torch.int8: {"atol": 7.5e-5},
        torch.uint8: {"atol": 7.5e-5},
    },
    "matmul": {
        torch.complex64: {"rtol": 5.2e-2},
    },
    "mm": {
        torch.complex64: {"rtol": 3.8e-2},
    },
    # Tolerance overrides justified for performance: native bfloat16/float16
    # pooling computations avoid costly intermediate upcasts to float32.
    "nn.functional.adaptive_avg_pool2d": {
        torch.bfloat16: {"rtol": 1.8e-2, "atol": 2.5e-2},
        torch.float16: {"rtol": 2.5e-3, "atol": 4.5e-3},
    },
    "nn.functional.adaptive_avg_pool3d": {
        torch.bfloat16: {"rtol": 1.8e-2, "atol": 2.0e-2},
        torch.float16: {"rtol": 3.5e-3, "atol": 3.0e-3},
    },
    "nn.functional.avg_pool2d": {
        torch.bfloat16: {"rtol": 3.1e-2, "atol": 4.5e-2},
        torch.float16: {"rtol": 3.1e-3, "atol": 4.1e-3},
    },
    "nn.functional.binary_cross_entropy": {
        torch.bfloat16: {"rtol": 9e-3, "atol": 5.9e-3},
        torch.float16: {"rtol": 8.5e-3, "atol": 4.4e-4},
        torch.float32: {"rtol": 1.8e-4, "atol": 6.9e-5},
    },
    "nn.functional.conv1d": {
        torch.float16: {"rtol": 8.2e-2},
        torch.float32: {"rtol": 9e-2},
    },
    "nn.functional.conv2d": {
        torch.bfloat16: {"atol": 1.5e-2},
        torch.float16: {"rtol": 3.9e-1, "atol": 4.2e-1},
        torch.float32: {"rtol": 4.2e-1, "atol": 4.6e-1},
    },
    "nn.functional.conv_transpose1d": {
        torch.bfloat16: {"atol": 8.3e-3},
        torch.float16: {"rtol": 6.7e-2, "atol": 1.6e-1},
        torch.float32: {"rtol": 7.1e-2, "atol": 2.1e-1},
    },
    "nn.functional.conv_transpose2d": {
        torch.bfloat16: {"rtol": 6.2e-2, "atol": 8e-2},
        torch.float16: {"rtol": 2.2e-1, "atol": 6.7e-2},
        torch.float32: {"rtol": 3e-1, "atol": 9.4e-2},
    },
    "nn.functional.cross_entropy": {
        torch.float16: {"rtol": 1.3e-3},
    },
    "nn.functional.embedding": {
        torch.float16: {
            "rtol": 3.9e-3,
            "atol": 2.4e-3,
        },  # TOLERANCE_OVERRIDE_OK
    },
    "nn.functional.embedding_bag": {
        torch.bfloat16: {"atol": 6.5e-2},
        torch.float16: {"atol": 2.5e-1},
    },
    "norm": {
        torch.complex64: {"rtol": 5.7e-6},
        torch.float32: {"rtol": 5.7e-6},
    },
    "polygamma": {
        torch.float32: {"rtol": 1.1e-5},
    },
    "pow": {
        torch.complex64: {"rtol": 7e-4, "atol": 3.8e-4},
    },
    "sin": {
        torch.complex64: {"rtol": 4.9e-6},
    },
    "sinh": {
        torch.complex64: {"rtol": 4.6e-6},
        torch.float32: {"rtol": 5.8e-6},
        torch.int16: {"rtol": 4e-6},
        torch.int32: {"rtol": 4e-6},
        torch.int64: {"rtol": 4e-6},
        torch.int8: {"rtol": 4e-6},
        torch.uint8: {"rtol": 4e-6},
    },
    "tan": {
        torch.complex64: {"rtol": 3.0e-5, "atol": 2.5e-4},
    },
    "tanh": {
        torch.complex64: {"rtol": 1.1e-5, "atol": 1.1e-5},
        torch.float32: {"atol": 3.8e-5},
        torch.int16: {"atol": 3.6e-5},
        torch.int32: {"atol": 3.6e-5},
        torch.int64: {"atol": 3.6e-5},
        torch.int8: {"atol": 3.6e-5},
        torch.uint8: {"atol": 3.6e-5},
    },
    "torch._scaled_mm": {
        torch.float8_e4m3fn: {"rtol": 9.4e-3, "atol": 4.7e-3},
    },
    # go/keep-sorted end
}  # end of ACCURACY_OVERRIDES_VS_GPU_COMPILED

# The GPU gradient tolerances are based on the TPU vs GPU forward pass tolerances.
ACCURACY_OVERRIDES_GRAD_VS_GPU: dict[
    str, dict[torch.dtype, dict[str, float]]
] = update_dict(
    copy.deepcopy(ACCURACY_OVERRIDES_VS_GPU),
    {
        # go/keep-sorted start
        "_foreach_erf": {
            torch.float16: {"rtol": 2.5e-3, "atol": 1e-4},
        },
        "_foreach_erfc": {
            torch.bfloat16: {"atol": 4.7e-3},
        },
        "_foreach_log10": {
            torch.float16: {"rtol": 1.2e-3, "atol": 5.9e-4},
        },
        "_foreach_pow": {
            torch.float16: {"rtol": 1.3e-3},
        },
        "_foreach_rsqrt": {
            torch.float32: {"rtol": 4e-6},
        },
        "_foreach_sigmoid": {
            torch.float16: {"atol": 5.9e-4},
        },
        "_native_batch_norm_legit": {
            torch.bfloat16: {"atol": 1.1e-2},
            torch.float16: {"atol": 1.4e-3},
        },
        "_thnn_fused_gru_cell": {
            torch.bfloat16: {"atol": 3.6e-3},
            torch.float16: {"atol": 1.1e-3},
            torch.float32: {"atol": 4.7e-5},
        },
        "_thnn_fused_lstm_cell": {
            torch.bfloat16: {"atol": 4.7e-3},
            torch.float16: {"atol": 5.2e-4},
            torch.float32: {"rtol": 6.5e-6, "atol": 5e-5},
        },
        "clamp": {
            torch.bfloat16: {"atol": 0.6},
            torch.float16: {"atol": 0.6},
        },
        "clamp_max": {
            torch.bfloat16: {"atol": 0.6},
        },
        "clamp_min": {
            torch.bfloat16: {"atol": 0.6},
        },
        "cumprod": {
            torch.bfloat16: {"rtol": 1.7e-1, "atol": 1.2e-2},
            torch.float16: {"rtol": 1.8e-2, "atol": 2e-3},
        },
        "einsum": {
            torch.float16: {"rtol": 2.4e-2},
            torch.float32: {"rtol": 2.3e-2},
        },
        "erf": {
            torch.bfloat16: {"rtol": 3.5e-2, "atol": 4.3e-4},
        },
        "erfinv": {
            torch.float16: {"rtol": 1.8e-3, "atol": 5.9e-3},
            torch.float32: {"rtol": 5e-5, "atol": 3.1e-3},
        },
        "fmax": {
            torch.bfloat16: {"atol": 0.6},
        },
        "fmin": {
            torch.bfloat16: {"atol": 0.6},
        },
        "kron": {
            torch.float16: {"rtol": 2.8e-3, "atol": 6.3e-2},
        },
        "ldexp": {
            torch.float32: {"rtol": 3.3e-6, "atol": 1.2e-3},
        },
        "lerp": {
            torch.bfloat16: {"rtol": 1.3e-1, "atol": 1.9e-1},
            torch.float16: {"rtol": 6.6e-3, "atol": 5.9e-3},
        },
        "linalg.inv": {
            torch.float32: {"rtol": 7.8e-2, "atol": 1.3e-2},
            torch.float64: {"rtol": 9e-6, "atol": 2.5e-7},
        },
        "linalg.lu": {
            torch.float32: {"rtol": 7.4e-1, "atol": 3.6e-2},
            torch.float64: {"rtol": 2.3e-5, "atol": 7.6e-6},
        },
        "linalg.lu_factor_ex": {
            torch.float32: {"rtol": 2.3e-5, "atol": 4.8e-4},
            torch.float64: {"rtol": 2.1e-5, "atol": 2e-6},
        },
        "linalg.lu_solve": {
            torch.float32: {"rtol": 4.7e-2, "atol": 1.4e-1},
        },
        "linalg.qr": {
            torch.float32: {"rtol": 9e-3, "atol": 1.5e-2},
        },
        "linalg.solve_ex": {
            torch.float32: {"rtol": 1.2e-1, "atol": 6e-2},
            torch.float64: {"rtol": 2.6e-6, "atol": 5.2e-6},
        },
        "linalg.vector_norm": {
            torch.complex64: {"rtol": 1.3e-5, "atol": 1.3e-5},
            torch.float16: {"rtol": 2.1e-3, "atol": 1.5e-3},
            torch.float32: {"rtol": 1.6e-5, "atol": 1.6e-5},
        },
        "log2": {
            torch.float16: {"rtol": 1.2e-3, "atol": 2.5e-4},
        },
        "log_softmax": {
            torch.bfloat16: {"rtol": 3.5e-1, "atol": 9.8e-3},
            torch.float16: {"rtol": 1.8e-2, "atol": 5.9e-3},
            torch.float32: {"rtol": 8e-4, "atol": 2.6e-4},
        },
        "logcumsumexp": {
            torch.bfloat16: {"rtol": 7.3e-2, "atol": 3.3e-2},
            torch.float16: {"rtol": 9.4e-3, "atol": 4.7e-3},
            torch.float32: {"rtol": 1.4e-4, "atol": 6.5e-5},
        },
        "native_batch_norm": {
            torch.bfloat16: {"atol": 1.1e-2},
            torch.float16: {"atol": 1.4e-3},
        },
        "nn.functional.cross_entropy": {
            torch.bfloat16: {"atol": 1.2e-3},
            torch.float16: {"atol": 3e-4},
        },
        "nn.functional.ctc_loss": {
            torch.float32: {"rtol": 3.6e-3, "atol": 2.2e-4},
        },
        "nn.functional.grid_sample": {
            torch.bfloat16: {"rtol": 1.9e-2, "atol": 1.9e-2},
            torch.float16: {"rtol": 3.6e-3, "atol": 2.4e-3},
        },
        "nn.functional.hardswish": {
            torch.bfloat16: {"rtol": 5.2e-2, "atol": 3e-3},
            torch.float16: {"rtol": 1.3e-2, "atol": 3.7e-4},
        },
        "nn.functional.interpolate": {
            torch.bfloat16: {"rtol": 2.7e-2, "atol": 7.9e-2},
            torch.float16: {"rtol": 1.8e-3, "atol": 4e-3},
        },
        "nn.functional.linear": {
            torch.float16: {"rtol": 3.6e-3, "atol": 5.9e-3},
            torch.float32: {"rtol": 4.1e-3, "atol": 2.5e-3},
        },
        "nn.functional.pdist": {
            torch.float32: {"rtol": 1.5e-5, "atol": 2.6e-5},
        },
        "nn.functional.scaled_dot_product_attention": {
            torch.bfloat16: {"rtol": 2.3e-1, "atol": 2.1e-1},
            torch.float16: {"rtol": 2.7e-1, "atol": 2.5e-1},
            torch.float32: {"rtol": 4.3e-1, "atol": 9.3e-1},
            torch.float64: {"rtol": 6e-6, "atol": 7.7e-6},
        },
        "nn.functional.softplus": {
            torch.float16: {"rtol": 1.5e-3, "atol": 1.3e-4},
        },
        "norm": {
            torch.complex64: {"rtol": 8.2e-6},
            torch.float16: {"atol": 1.2e-3},
            torch.float32: {"rtol": 8.8e-6},
        },
        "polygamma": {
            torch.bfloat16: {"rtol": 3.6e-2},
            torch.float32: {"rtol": 1.1e-5},
        },
        "prod": {
            torch.float16: {"rtol": 1.6e-3, "atol": 9.8e-4},
        },
        "std": {
            torch.float16: {"atol": 6e-4},
        },
        "tan": {
            torch.float32: {"rtol": 3.8e-6, "atol": 8.4e-3},
        },
        # go/keep-sorted end
    },
)
if test_utils.parse_version(torch.__version__) < (2, 14, 0):
  # PyTorch 2.14+ changed clamp_min/clamp_max derivative tie-breaking at
  # self == bound from 1.0 to 0.0/0.5 ( matching the PyTorch 2.15 GPU goldens).
  # In OSS (which pins PyTorch 2.11), autograd decomposition of
  # _foreach_clamp_min/_foreach_clamp_max uses the pre-2.14 formula.
  update_dict(
      ACCURACY_OVERRIDES_GRAD_VS_GPU,
      {
          # go/keep-sorted start
          "_foreach_clamp_max": {
              torch.bfloat16: {"atol": 1.0},
              torch.float16: {"atol": 1.0},
          },
          "_foreach_clamp_min": {
              torch.bfloat16: {"atol": 1.0},
              torch.float16: {"atol": 1.0},
          },
          # go/keep-sorted end
      },
  )
