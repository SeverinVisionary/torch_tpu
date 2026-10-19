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

"""Tests for error handling in Pallas TPU kernels and custom ops."""

from __future__ import annotations

import enum
import os
from typing import Sequence

from absl.testing import absltest
import jax
import jax.numpy as jnp
import packaging.version
import torch
from torch_tpu._internal import pallas
from torch_tpu._internal import testing as tt_testing
from tests import seed_test_utils


@pallas.jax_op("test_pallas_errors::pass_through")
def pass_through(x: jax.Array) -> jax.Array:
  return x


def add_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  return x + y


class MyCustomType:
  pass


class MyStrEnum(enum.StrEnum):
  MODE_A = "mode_a"
  MODE_B = "mode_b"


class MyEnum(enum.Enum):
  OPT_1 = 1
  OPT_2 = 2


class PallasKernelErrorsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.device("tpu")

  def _skip_if_fp4_unsupported(self) -> None:
    accelerator = torch.accelerator.current_accelerator()
    if (
        accelerator is None
        or accelerator.type != "tpu"
        or not torch.accelerator.is_available()
    ):
      self.skipTest("FP4 requires a TPU generation supported by TorchTPU")

  def test_symbolic_shape_raises_error(self):
    symbolic_shape = torch.fx.experimental.symbolic_shapes
    shape_env = symbolic_shape.ShapeEnv()

    x = torch.randn(5, device=self.device)
    with torch._subclasses.FakeTensorMode(shape_env=shape_env) as mode:
      fake_x = mode.from_tensor(
          x,
          symbolic_context=symbolic_shape.StatelessSymbolicContext(
              dynamic_sizes=[symbolic_shape.DimDynamic.DYNAMIC]
          ),
      )
      with self.assertRaisesRegex(
          RuntimeError, "Symbolic dimensions are not supported"
      ):
        pass_through(fake_x)

  def test_inconsistent_donate_and_aliases_raises_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "donate_argnums must be None or the same as the keys of"
        " input_output_aliases.",
    ):
      pallas.custom_jax_kernel(
          add_vectors_jax,
          name="add_vectors",
          input_output_aliases={0: 0},
          donate_argnums=(1,),
      )

  def test_custom_call_export_compilation_failure(self):
    is_oss = os.getenv("IS_OSS", "0") == "1"
    jax_version = packaging.version.Version(jax.__version__)
    if is_oss and jax_version < packaging.version.Version("0.10.3"):
      self.skipTest("Jax version doesn't support custom_call export.")

    def custom_call_fn(x: jax.Array) -> jax.Array:
      return jax.ffi.ffi_call(
          "test_custom_call", jax.ShapeDtypeStruct(x.shape, x.dtype)
      )(x)

    custom_call_op = pallas.jax_op("test_errors::custom_call", custom_call_fn)

    # Check that it successfully reaches XLA and raises an error.
    with self.assertRaisesRegex(
        RuntimeError,
        expected_regex="""failed to compile MLIR module -- the XLA compiler failed with: custom emitter for test_custom_call not found""",
    ):
      x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
      y = custom_call_op(x)
      y.cpu()
    tt_testing.reset_eager_state()

  def test_fp4_packed_dimension_non_contiguous_raises_error(self):
    self._skip_if_fp4_unsupported()

    u8_t = torch.zeros((8, 256), dtype=torch.uint8)
    fp4_t = u8_t.view(torch.float4_e2m1fn_x2).to("tpu")
    fp4_t_transposed = fp4_t.t()
    self.assertFalse(fp4_t_transposed.is_contiguous())

    def fn_identity(x: jax.Array) -> jax.Array:
      return x

    op = pallas.jax_op("test_errors::identity_error", fn_identity)

    with self.assertRaisesRegex(
        ValueError,
        r"expected packed dimension of float4_e2m1fn_x2 tensor to be "
        r"contiguous \(stride 1\)",
    ):
      op(fp4_t_transposed)


class PallasSignatureErrorsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def test_invalid_base_type_rejected(self):
    # dict is not supported as argument.
    def invalid_fn_dict(x: jax.Array, y: dict[str, int]) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(
        ValueError, r"Arguments at indices \[1\] are invalid"
    ):
      pallas.jax_op("test_errors::invalid_dict", invalid_fn_dict)

    # Custom classes are not supported.
    def invalid_fn_custom(x: jax.Array, y: MyCustomType) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(
        ValueError, r"Arguments at indices \[1\] are invalid"
    ):
      pallas.jax_op("test_errors::invalid_custom", invalid_fn_custom)

    # Sequence of str is not supported in PyTorch custom op schemas.
    def invalid_fn_seq_str(x: jax.Array, y: Sequence[str]) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(
        ValueError, r"Arguments at indices \[1\] are invalid"
    ):
      pallas.jax_op("test_errors::invalid_seq_str", invalid_fn_seq_str)

    # Sequence of enum is not supported in PyTorch custom op schemas.
    def invalid_fn_seq_enum(x: jax.Array, y: Sequence[MyStrEnum]) -> jax.Array:
      del y  # Unused.
      return x

    with self.assertRaisesRegex(
        ValueError, r"Arguments at indices \[1\] are invalid"
    ):
      pallas.jax_op("test_errors::invalid_seq_enum", invalid_fn_seq_enum)

  def test_invalid_return_type_rejected(self):
    """Verifies unsupported return types are rejected."""

    # MyCustomType is not supported as return type.
    def invalid_return_fn(x: jax.Array) -> MyCustomType:
      del x  # Unused.
      return MyCustomType()

    with self.assertRaisesRegex(ValueError, "The return annotation is invalid"):
      pallas.jax_op("test_errors::invalid_return", invalid_return_fn)

    # String is not supported as return type.
    def fn_return_str(x: jax.Array) -> str:
      del x  # Unused.
      return "foo"

    with self.assertRaisesRegex(ValueError, "The return annotation is invalid"):
      pallas.jax_op("test_errors::fn_return_str", fn_return_str)

    # Dtype is not supported as return type.
    def fn_return_dtype(x: jax.Array) -> jnp.dtype:
      del x  # Unused.
      return jnp.float32

    with self.assertRaisesRegex(ValueError, "The return annotation is invalid"):
      pallas.jax_op("test_errors::fn_return_dtype", fn_return_dtype)

    # Enum is not supported as return type.
    def fn_return_enum(x: jax.Array) -> MyEnum:
      del x  # Unused.
      return MyEnum.OPT_1

    with self.assertRaisesRegex(ValueError, "The return annotation is invalid"):
      pallas.jax_op("test_errors::fn_return_enum", fn_return_enum)

  def test_invalid_jitted_function_rejected(self):
    """Verifies invalid jitted functions are rejected by pallas.jax_op."""

    def my_fn(x: MyCustomType) -> MyCustomType:
      return x

    jitted = jax.jit(my_fn)
    with self.assertRaisesRegex(
        ValueError, r"Arguments at indices \[0\] are invalid"
    ):
      pallas.jax_op("test_errors::jitted_invalid", jitted)

  def test_unannotated_arguments_rejected(self):
    """Verifies unannotated arguments are rejected."""

    def unannotated_fn(x) -> jax.Array:
      return x

    with self.assertRaisesRegex(
        ValueError, "Missing argument type annotation for JAX function"
    ):
      pallas.jax_op("test_errors::unannotated", unannotated_fn)


if __name__ == "__main__":
  absltest.main()
