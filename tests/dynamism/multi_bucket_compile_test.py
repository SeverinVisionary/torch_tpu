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

"""Tests for multi-bucket bounded dynamism compilation and lookahead prefetching."""

from collections.abc import Iterator
import contextlib
import copy
import dataclasses
from unittest import mock

from absl import logging
from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.compile import torch_tpu_compiled_executable
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


@dataclasses.dataclass(frozen=True)
class _StaticCompileRecord:
  """Bounded input shapes and dynamic input layouts of one static compile."""

  bounded_shapes: tuple[tuple[int, ...], ...]
  argument_layouts: tuple[tuple[int, ...], ...]


@contextlib.contextmanager
def _record_static_compiles() -> Iterator[list[_StaticCompileRecord]]:
  """Records every StaticCompiler call, i.e. every static or bucket compile."""
  records: list[_StaticCompileRecord] = []
  orig_static_call = _backend.compiler.StaticCompiler.__call__

  def wrapped_static_call(static_self, gm, example_inputs, *args, **kwargs):
    bounds = kwargs.get("bounds") or [None] * len(example_inputs)
    records.append(
        _StaticCompileRecord(
            bounded_shapes=tuple(
                tuple(inp.shape)
                for inp, bound in zip(example_inputs, bounds)
                if bound is not None
            ),
            argument_layouts=tuple(
                tuple(layout)
                for layout in kwargs.get("argument_layouts") or []
                if layout
            ),
        )
    )
    return orig_static_call(static_self, gm, example_inputs, *args, **kwargs)

  with mock.patch.object(
      _backend.compiler.StaticCompiler,
      "__call__",
      autospec=True,
      side_effect=wrapped_static_call,
  ):
    yield records


@dataclasses.dataclass(frozen=True)
class _ShapeEnvGuards:
  """Dynamo shape guards before and after one dynamic (bucketed) compile."""

  before: tuple[str, ...]
  after: tuple[str, ...]


@contextlib.contextmanager
def _record_shape_env_guards() -> Iterator[list[_ShapeEnvGuards]]:
  """Records the ShapeEnv guards around every DynamicCompiler call."""
  records: list[_ShapeEnvGuards] = []
  orig_dynamic_call = _backend.dynamic_compiler.DynamicCompiler.__call__

  def guard_exprs(example_inputs) -> tuple[str, ...]:
    for arg in example_inputs:
      if isinstance(arg, torch.SymInt):
        return tuple(str(g.expr) for g in arg.node.shape_env.guards)
    return ()

  def wrapped_dynamic_call(dynamic_self, gm, example_inputs, *args, **kwargs):
    before = guard_exprs(example_inputs)
    result = orig_dynamic_call(
        dynamic_self, gm, example_inputs, *args, **kwargs
    )
    records.append(_ShapeEnvGuards(before, guard_exprs(example_inputs)))
    return result

  with mock.patch.object(
      _backend.dynamic_compiler.DynamicCompiler,
      "__call__",
      autospec=True,
      side_effect=wrapped_dynamic_call,
  ):
    yield records


class MultiBucketCompilationTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  def test_multi_bucket_lookahead_async_compilation(self):
    def fn(x):
      return x + 1.0

    orig_static_call = _backend.compiler.StaticCompiler.__call__
    orig_resolve = torch_tpu_compiled_executable.AsyncCompiledArtifact.resolve
    compiled_upper_bounds = []
    bounds_at_first_resolve = []

    def wrapped_static_call(static_self, gm, example_inputs, *args, **kwargs):
      for inp in example_inputs:
        if isinstance(inp, torch.Tensor) and inp.dim() == 1:
          compiled_upper_bounds.append(inp.shape[0])
          break
      return orig_static_call(static_self, gm, example_inputs, *args, **kwargs)

    def wrapped_resolve(artifact_self):
      if not bounds_at_first_resolve:
        bounds_at_first_resolve.extend(compiled_upper_bounds)
      return orig_resolve(artifact_self)

    with (
        mock.patch.object(
            _backend.dynamic_compiler,
            "DynamicCompiler",
            wraps=_backend.dynamic_compiler.DynamicCompiler,
        ) as mock_dyn_compiler,
        mock.patch.object(
            _backend.compiler.StaticCompiler,
            "__call__",
            autospec=True,
            side_effect=wrapped_static_call,
        ) as mock_static_call,
        mock.patch.object(
            torch_tpu_compiled_executable.AsyncCompiledArtifact,
            "resolve",
            autospec=True,
            side_effect=wrapped_resolve,
        ),
    ):
      compiled = torch.compile(
          fn,
          backend="tpu",
          options={
              "num_dynamic_buckets": 4,
          },
      )

      # Step 1: Execute in Bucket 0 ([2, 128]).
      # By default num_lookahead_buckets=num_dynamic_buckets=4 compiles
      # Bucket 0 (128) and prefetches Buckets 1 (256), 2 (384), and 3 (512)
      # BEFORE resolving Bucket 0.
      x0 = torch.ones(32, device=self.device)
      torch._dynamo.mark_dynamic(x0, 0, min=2, max=512)
      out0 = compiled(x0)
      utils.assert_close(out0, fn(x0))
      self.assertEqual(mock_dyn_compiler.call_count, 1)
      self.assertEqual(mock_static_call.call_count, 4)
      self.assertEqual(compiled_upper_bounds, [128, 256, 384, 512])
      self.assertEqual(bounds_at_first_resolve, [128, 256, 384, 512])

      # Step 2: Another shape in Bucket 0 ([2, 128]) -> no new compilations.
      x0_again = torch.ones(64, device=self.device)
      out0_again = compiled(x0_again)
      utils.assert_close(out0_again, fn(x0_again))
      self.assertEqual(mock_static_call.call_count, 4)

      # Step 3: Execute in Buckets 1 ([129, 256]), 2 ([257, 384]), and 3 ([385, 512]).
      # Reuses prefetched buckets; no further compilations needed.
      for length in (160, 300, 420):
        x_next = torch.ones(length, device=self.device)
        out_next = compiled(x_next)
        utils.assert_close(out_next, fn(x_next))
      self.assertEqual(mock_dyn_compiler.call_count, 1)
      self.assertEqual(mock_static_call.call_count, 4)
      self.assertEqual(compiled_upper_bounds, [128, 256, 384, 512])

  def test_multi_bucket_custom_bucket_size(self):
    def fn(x):
      return x + 1.0

    orig_static_call = _backend.compiler.StaticCompiler.__call__
    compiled_upper_bounds = []

    def wrapped_static_call(static_self, gm, example_inputs, *args, **kwargs):
      for inp in example_inputs:
        if isinstance(inp, torch.Tensor) and inp.dim() == 1:
          compiled_upper_bounds.append(inp.shape[0])
          break
      return orig_static_call(static_self, gm, example_inputs, *args, **kwargs)

    with mock.patch.object(
        _backend.compiler.StaticCompiler,
        "__call__",
        autospec=True,
        side_effect=wrapped_static_call,
    ) as mock_static_call:
      compiled = torch.compile(
          fn,
          backend="tpu",
          options={
              "num_dynamic_buckets": 3,
              "bucket_size": 64,
          },
      )

      x0 = torch.ones(16, device=self.device)
      torch._dynamo.mark_dynamic(x0, 0, min=2, max=192)
      out0 = compiled(x0)
      utils.assert_close(out0, fn(x0))
      self.assertEqual(mock_static_call.call_count, 3)
      self.assertEqual(compiled_upper_bounds, [64, 128, 192])

      for length in (96, 160):
        x_next = torch.ones(length, device=self.device)
        out_next = compiled(x_next)
        utils.assert_close(out_next, fn(x_next))
      self.assertEqual(mock_static_call.call_count, 3)

  def test_bucket_switch_keeps_capacity_alignment_of_clamped_symbol(self):
    # GQA decode step with one bounded KV length symbol per layer (mark_dynamic
    # gives every layer its own symbol, as in HF-style caches): the caches grow
    # by torch.cat and layer 0's length slices two static buffers of size
    # max_len, which refines its range to [2, max_len - 1] = 2053, an
    # unaligned cap. Bucket 1 must give layer 0 the same `128k - 1` capacity
    # (2175) as layer 1: XLA picks the on-device layout of a bounded output
    # from its bound, so a 2053 capacity emits layer 0's KV in a different
    # layout than the one the lookahead executable of bucket 1 was compiled
    # for, and the step after the bucket switch recompiles the whole bucket.
    prefill_len, num_steps = 2044, 10
    max_len = prefill_len + num_steps
    num_layers = 2

    class DecodeStep(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.rope = torch.nn.Buffer(
            torch.randn(1, 1, max_len, 128, dtype=torch.bfloat16)
        )
        # Additive mask hiding every odd key position.
        mask = torch.zeros(1, 1, 1, max_len, dtype=torch.bfloat16)
        mask[..., 1::2] = -1e4
        self.mask = torch.nn.Buffer(mask)

      def forward(self, q, k_new, v_new, *caches):
        past_len = caches[0].shape[2]
        kv_len = past_len + 1
        # Static buffers sliced by the dynamic cache length (dynamic start and
        # dynamic end) refine `past_len` to <= max_len - 1.
        k_new = k_new + self.rope[:, :, past_len:kv_len, :]
        mask = self.mask[:, :, :, :kv_len]
        out = torch.zeros_like(q)
        new_caches = []
        for layer, (k, v) in enumerate(zip(caches[::2], caches[1::2])):
          k = torch.cat([k, k_new], dim=2)
          v = torch.cat([v, v_new], dim=2)
          out = out + torch.nn.functional.scaled_dot_product_attention(
              q, k, v, attn_mask=mask if layer == 0 else None, enable_gqa=True
          )
          new_caches.extend((k, v))
        return out, *new_caches

    model = DecodeStep()
    compiled = torch.compile(
        copy.deepcopy(model).to(self.device),
        backend="tpu",
        options={"num_dynamic_buckets": 2},
    )

    q = torch.randn(1, 4, 1, 128, dtype=torch.bfloat16)
    caches = []
    for layer in range(num_layers):
      k = torch.randn(1, 2, prefill_len, 128, dtype=torch.bfloat16)
      v = torch.randn(1, 2, prefill_len, 128, dtype=torch.bfloat16)
      if layer == 0:
        # Hidden by the mask; a wrong mask slice shows up as a large output.
        v[:, :, 1::2, :] = 8.0
      caches.extend((k, v))
    device_caches = [c.to(self.device) for c in caches]
    for c in device_caches:
      torch._dynamo.mark_dynamic(c, 2)

    first_bucket_upper = 2047  # 128 * 16 - 1 (outputs have length s + 1)
    compiles_after_switch = None
    with (
        _record_static_compiles() as compiles,
        _record_shape_env_guards() as guards,
    ):
      for step in range(num_steps):
        past_len = prefill_len + step
        k_new = torch.randn(1, 2, 1, 128, dtype=torch.bfloat16)
        v_new = torch.randn(1, 2, 1, 128, dtype=torch.bfloat16)
        expected = model(q, k_new, v_new, *caches)
        actual = compiled(
            q.to(self.device),
            k_new.to(self.device),
            v_new.to(self.device),
            *device_caches,
        )
        torch.accelerator.synchronize()
        layouts = [
            torch.tpu.get_device_layout_if_materialized(c)[0]
            for c in actual[1:]
        ]
        logging.info(
            "step %d: past_len=%d, compiles=%d, fed-back KV layouts=%s",
            step,
            past_len,
            len(compiles),
            layouts,
        )
        utils.assert_close(actual[0].cpu(), expected[0], rtol=2e-2, atol=2e-2)
        for actual_cache, expected_cache in zip(actual[1:], expected[1:]):
          utils.assert_close(actual_cache.cpu(), expected_cache)
        caches = list(expected[1:])
        device_caches = list(actual[1:])
        if past_len == first_bucket_upper + 1:
          compiles_after_switch = len(compiles)

    logging.info("shape_env guards: %s", guards)
    # Besides the bound checks of the unbounded symbols on the last bucket
    # (`s <= 2175`, which makes Dynamo recompile past it), the dynamic compile
    # installs no guard: the clamped symbol's aligned capacity is implied by
    # its range `s <= 2053`.
    self.assertLen(guards, 1)
    added_guards = guards[0].after[len(guards[0].before) :]
    self.assertNotEmpty(added_guards)
    for guard in added_guards:
      self.assertRegex(guard, r"^s\d+ <= 2175$")
    # Bucket 0 is [2, 2047] and bucket 1 [2048, 2175] for every symbol; the
    # clamped symbol must not get the unaligned 2053 capacity.
    bucket_capacities = {
        tuple(sorted({shape[2] for shape in record.bounded_shapes}))
        for record in compiles
    }
    self.assertEqual(bucket_capacities, {(2047,), (2175,)})
    # The first step in bucket 1 runs the lookahead executable and the steps
    # after it reuse it: no compile after the bucket switch.
    self.assertIsNotNone(compiles_after_switch)
    self.assertEqual(len(compiles), compiles_after_switch)

  def test_bucket_switch_with_unaligned_mark_dynamic_max(self):
    # `mark_dynamic(min=2, max=200)` is a StrictMinMaxConstraint: the second
    # bucket extends to the aligned capacity 255 (>= 200) instead of 200, and
    # the bound checks on that capacity must not add a guard (Dynamo would
    # raise a ConstraintViolationError for a guard narrowing the range).
    def fn(x, y):
      return (
          torch.cat([x, y]) + 1.0
      )  # Output length s + 1 -> capacities 128k - 1.

    with (
        _record_static_compiles() as compiles,
        _record_shape_env_guards() as guards,
    ):
      compiled = torch.compile(
          fn, backend="tpu", options={"num_dynamic_buckets": 2}
      )
      for length in (100, 127, 128, 129, 200):
        x = torch.ones(length, device=self.device)
        y = torch.ones(1, device=self.device)
        if length == 100:
          torch._dynamo.mark_dynamic(x, 0, min=2, max=200)
        utils.assert_close(compiled(x, y), fn(x, y))

    self.assertLen(guards, 1)
    self.assertEqual(guards[0].before, guards[0].after)
    self.assertEqual(
        [record.bounded_shapes for record in compiles], [((127,),), ((255,),)]
    )


if __name__ == "__main__":
  absltest.main()
