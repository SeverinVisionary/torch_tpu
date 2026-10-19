# Copyright 2026 Google LLC
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

"""Consolidated error tests for compiler components."""

import concurrent.futures
import os
from unittest import mock

from absl.testing import absltest
import torch
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal import execution_mode
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import compiler as tpu_compiler_mod
from torch_tpu._internal.compile import torch_tpu_compiled_executable
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.fx_passes import sink_get_attr_constants
from tests import seed_test_utils

AsyncCompiledArtifact = torch_tpu_compiled_executable.AsyncCompiledArtifact
GraphModule = fx.GraphModule
Graph = fx.Graph


class AsyncCompiledArtifactErrorsTest(seed_test_utils.RepeatableTest):

  def test_magic_attributes_do_not_resolve_and_raise_attribute_error(self):
    future = concurrent.futures.Future()
    artifact = AsyncCompiledArtifact(future)

    with self.assertRaises(AttributeError):
      _ = artifact.__non_existent_magic_attr__

    self.assertFalse(artifact.is_resolved)

    # Special framework attributes return class defaults without resolving
    self.assertIsNone(artifact._fx_graph_cache_key)
    self.assertFalse(artifact._is_fallback)
    self.assertFalse(artifact.is_resolved)

  def test_exception_propagation_on_resolve(self):
    future = concurrent.futures.Future()
    future.set_exception(RuntimeError("Simulated XLA compilation failure"))

    artifact = AsyncCompiledArtifact(future)
    self.assertFalse(artifact.is_resolved)

    with self.assertRaisesRegex(
        RuntimeError, "Simulated XLA compilation failure"
    ):
      artifact.resolve()

    self.assertFalse(artifact.is_resolved)

  def test_exception_propagation_on_call(self):
    future = concurrent.futures.Future()
    future.set_exception(ValueError("Invalid tensor layout"))

    artifact = AsyncCompiledArtifact(future)

    with self.assertRaisesRegex(ValueError, "Invalid tensor layout"):
      artifact([])

    self.assertFalse(artifact.is_resolved)

  def test_exception_propagation_on_getattr(self):
    future = concurrent.futures.Future()
    future.set_exception(RuntimeError("Compile error"))

    artifact = AsyncCompiledArtifact(future)

    with self.assertRaisesRegex(RuntimeError, "Compile error"):
      _ = artifact.some_missing_attr


class EnsureNoDynamicErrorsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    self.enterContext(
        mock.patch.dict(
            os.environ,
            {"TORCHDYNAMO_VERBOSE": "1", "TORCH_LOGS": "+dynamo"},
        )
    )
    self.addCleanup(torch.compiler.reset)

  def test_dynamic_true_not_supported(self):
    def simple(x):
      return 0.3 * x

    input_1 = torch.tensor([0.1, 0.2], device=torch.device("tpu"))

    compiled = torch.compile(simple, dynamic=True, backend="tpu")
    with self.assertRaises(Exception) as err:
      compiled(input_1).to("cpu")
    self.assertIn("dynamic=True", str(err.exception))


class SinkGetAttrConstantsErrorsTest(seed_test_utils.RepeatableTest):

  def test_placeholder_arg_count_mismatch_raises_value_error(self):
    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_c = sub_graph.placeholder("c")
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_c))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    # Submodule has 2 placeholders but the call site only passes 1 argument.
    parent_graph = Graph()
    p_x = parent_graph.placeholder("x")
    call_submod = parent_graph.call_module("submod_0", (p_x,))
    parent_graph.output(call_submod)

    parent_root = torch.nn.Module()
    parent_root.submod_0 = submod
    parent_gm = GraphModule(parent_root, parent_graph)

    with self.assertRaises(ValueError):
      sink_get_attr_constants.apply(parent_gm)


class AsyncCompileErrorsTest(seed_test_utils.RepeatableTest):

  def test_async_compile_error_propagation(self):
    """Verifies that background compilation failures surface cleanly during resolution."""

    class SimpleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

      def forward(self, x):
        return self.fc(x)

    model = SimpleModel().to("tpu")

    def failing_compile(*args, **kwargs):
      raise RuntimeError("Simulated XLA lowering failure")

    with mock.patch.object(
        tpu_compiler_mod.StaticCompiler, "_compile", failing_compile
    ):
      compiled = torch.compile(
          torch.no_grad()(model),
          backend="tpu",
          fullgraph=True,
          dynamic=False,
          options={"async_compile": True},
      )

      with self.assertRaises(tpu_compile.AsyncCompilationSubmitted) as cm:
        compiled(torch.ones(4, 4, device="tpu"))

      art = cm.exception.artifact
      self.assertIsNotNone(art)

      with self.assertRaisesRegex(
          RuntimeError, "Simulated XLA lowering failure"
      ):
        art.resolve()


class CompiledRngErrorsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    self.device = torch.device("tpu")

  def test_compiled_generator_arg_triggers_graph_break(self):
    """Verifies passing custom generator triggers graph break."""
    g = torch.Generator(device=self.device)
    g.manual_seed(42)
    x = torch.zeros(10, device=self.device)

    def fn_pass_gen(x, gen):
      return torch.rand_like(x, generator=gen)

    with self.assertRaises(torch._dynamo.exc.Unsupported):
      torch.compile(fn_pass_gen, fullgraph=True)(x, g)

  def test_compiled_manual_seed_inside_fn_triggers_graph_break(self):
    """Verifies seeding inside compiled function triggers graph break."""
    x = torch.zeros(10, device=self.device)

    def fn_manual_seed(x):
      torch.manual_seed(42)
      return torch.rand_like(x)

    with self.assertRaises(torch._dynamo.exc.Unsupported):
      torch.compile(fn_manual_seed, fullgraph=True)(x)

  def test_compiled_generator_instantiation_inside_fn_triggers_graph_break(
      self,
  ):
    """Verifies using generator in compiled function triggers graph break."""
    x = torch.zeros(10, device=self.device)

    def fn_create_gen(x):
      g = torch.Generator(device=self.device)
      return torch.rand_like(x, generator=g)

    with self.assertRaises(torch._dynamo.exc.Unsupported):
      torch.compile(fn_create_gen, fullgraph=True)(x)

  def test_compiled_graphsafe_get_state_triggers_graph_break(self):
    """Verifies graphsafe_get_state triggers a graph break in torch.compile."""
    x = torch.zeros(10, device=self.device)
    g = torch.Generator(device=self.device)

    def fn_get_state(x):
      _ = g.graphsafe_get_state()
      return torch.rand_like(x)

    with self.assertRaises(torch._dynamo.exc.Unsupported):
      torch.compile(fn_get_state, fullgraph=True)(x)

  def test_compiled_graphsafe_set_state_triggers_graph_break(self):
    """Verifies graphsafe_set_state triggers a graph break in torch.compile."""
    x = torch.zeros(10, device=self.device)
    g = torch.Generator(device=self.device)
    saved_state = g.clone_state()

    def fn_set_state(x):
      g.graphsafe_set_state(saved_state)
      return torch.rand_like(x)

    with self.assertRaises(torch._dynamo.exc.Unsupported):
      torch.compile(fn_set_state, fullgraph=True)(x)


def _eager_mode_compile_fx_graph():
  return execution_mode.set_eager_mode(
      execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
  )


class CompileApiErrorsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    self.device = torch.device("tpu")
    tt_testing.reset_eager_state()

  def test_compilation_with_forced_layout_mismatch_raises(self):
    with _eager_mode_compile_fx_graph():
      x2 = torch.ones(2, 3, device="cpu").to(device=self.device)
      z2 = x2 + x2

    compile_result_fail = tpu_torch_compile.traverse_and_compile(
        [z2], [x2], argument_layouts=[[0, 1]]
    )
    self.assertIsNotNone(compile_result_fail.executable)

    input_tensor = torch.ones(2, 3, device=self.device)
    input_tensor.cpu()  # Force materialization

    results_fail = tpu_torch_compile.execute(
        compile_result_fail.executable, [input_tensor]
    )
    self.assertLen(results_fail, 1)

    with self.assertRaises(RuntimeError) as err:
      results_fail[0].cpu()

    self.assertIn("layout", str(err.exception).lower())
    tt_testing.reset_eager_state()

  def test_static_compiler_buffer_donation_raises_on_deleted_buffer(self):
    def simple_fn(x, y):
      return x + y, x * 2

    x = torch.randn(3, 5, device=self.device)
    y = torch.randn(3, 5, device=self.device)

    compiled_executable = tpu_compiler_mod.StaticCompiler()(
        make_fx(simple_fn)(x, y), [x, y], donated_inputs=[0]
    )

    res = compiled_executable(x, y)
    res[0].cpu()  # Wait for execution

    with self.assertRaisesRegex(
        RuntimeError, "INVALID_ARGUMENT: Buffer has been deleted or donated"
    ):
      x.cpu()


if __name__ == "__main__":
  absltest.main()
