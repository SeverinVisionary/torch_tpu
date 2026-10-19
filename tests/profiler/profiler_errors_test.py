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

"""Consolidated error tests for TorchTPU profiler components."""

import os
import pathlib
import shutil
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import portpicker
import torch
from torch_tpu._internal import profiler
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.profiler import _impl as profiler_impl
from torch_tpu._internal.profiler import _profiler_backend as profiler_backend
from torch_tpu._internal.profiler.profiler_config import TpuProfilerConfig
from tests import seed_test_utils

# Set TMPDIR to TEST_TMPDIR so TorchTPU's profiler writes artifacts into the
# test's temporary directory instead of falling back to system /tmp.
if "TEST_TMPDIR" in os.environ:
  os.environ["TMPDIR"] = os.environ["TEST_TMPDIR"]


def _cleanup_profile_dir() -> None:
  tmpdir = os.environ.get("TEST_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
  profile_dir = pathlib.Path(tmpdir) / "plugins" / "profile"
  if profile_dir.exists():
    shutil.rmtree(profile_dir)


class ProfilerApiErrorsTest(seed_test_utils.RepeatableTest):

  def test_profile_api_with_invalid_rank(self):
    output_dir = self.create_tempdir("profile_api_with_invalid_rank").full_path
    handler = profiler.xprof_trace_handler(
        dir_name=output_dir, worker_rank="../../invalid"
    )
    with self.assertRaisesRegex(
        ValueError, "worker_rank cannot contain path separators"
    ):
      with profiler.profile(
          activities=[profiler.ProfilerActivity.CPU],
          on_trace_ready=handler,
      ):
        pass

  def test_profile_api_gpu_unsupported(self):
    output_dir = self.create_tempdir("profile_api_gpu_unsupported").full_path
    handler = profiler.xprof_trace_handler(dir_name=output_dir)
    with self.assertRaisesRegex(ValueError, "GPU profiling is not supported."):
      with profiler.profile(
          activities=[profiler.ProfilerActivity.GPU],
          on_trace_ready=handler,
      ):
        pass


class ProfilerBackendErrorsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.tmpdir = self.create_tempdir().full_path
    try:
      profiler_backend.stop_profiler_server()
    except RuntimeError:
      pass
    try:
      profiler_backend.stop_trace(os.path.join(self.tmpdir, "unused.pb"))
    except RuntimeError:
      pass

  def test_double_start_server_fails(self):
    port = portpicker.pick_unused_port()
    profiler_backend.start_profiler_server(port)
    try:
      with self.assertRaisesRegex(
          RuntimeError, "server has already been started"
      ):
        profiler_backend.start_profiler_server(port)
    finally:
      profiler_backend.stop_profiler_server()

  def test_double_start_trace_fails(self):
    profiler_backend.start_trace(self.tmpdir)
    try:
      with self.assertRaisesRegex(
          RuntimeError, "session has already been started"
      ):
        profiler_backend.start_trace(self.tmpdir)
    finally:
      profiler_backend.stop_trace(os.path.join(self.tmpdir, "trace.pb"))

  def test_stop_server_without_start_fails(self):
    with self.assertRaisesRegex(RuntimeError, "server has not been started"):
      profiler_backend.stop_profiler_server()

  def test_stop_trace_without_start_fails(self):
    with self.assertRaisesRegex(RuntimeError, "session has not been started"):
      profiler_backend.stop_trace(os.path.join(self.tmpdir, "trace.pb"))

  def test_stop_server_twice_fails(self):
    port = portpicker.pick_unused_port()
    profiler_backend.start_profiler_server(port)
    profiler_backend.stop_profiler_server()
    with self.assertRaisesRegex(RuntimeError, "server has not been started"):
      profiler_backend.stop_profiler_server()

  def test_stop_trace_twice_fails(self):
    profiler_backend.start_trace(self.tmpdir)
    profiler_backend.stop_trace(os.path.join(self.tmpdir, "trace.pb"))
    with self.assertRaisesRegex(RuntimeError, "session has not been started"):
      profiler_backend.stop_trace(os.path.join(self.tmpdir, "trace.pb"))

  def test_retry_start_after_stop_failure(self):
    profiler_backend.start_trace(self.tmpdir)
    invalid_filename = self.tmpdir
    with self.assertRaisesRegex(RuntimeError, "failed to create file"):
      profiler_backend.stop_trace(invalid_filename)

    profiler_backend.start_trace(self.tmpdir)
    profiler_backend.stop_trace(os.path.join(self.tmpdir, "trace.pb"))


class ProfilerImplErrorsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    profiler_impl._profile_state.reset()
    if profiler_impl._profiler_server is not None:
      profiler_impl.stop_server()
    if profiler_impl._profile_state.profile_session is not None:
      profiler_impl.stop_trace()

  def tearDown(self):
    if profiler_impl._profiler_server is not None:
      profiler_impl.stop_server()
    if profiler_impl._profile_state.profile_session is not None:
      profiler_impl.stop_trace()
    super().tearDown()

  def test_double_start_server_fails(self):
    server = None
    port1 = portpicker.pick_unused_port()
    try:
      server = profiler_impl.start_server(port1)
      self.assertIsNotNone(server)
      with self.assertRaises(ValueError):
        profiler_impl.start_server(port1)
    finally:
      if server is not None:
        profiler_impl.stop_server()

  def test_double_start_server_diff_ports_fails(self):
    server = None
    port1 = portpicker.pick_unused_port()
    port2 = portpicker.pick_unused_port()
    try:
      server = profiler_impl.start_server(port1)
      self.assertIsNotNone(server)
      with self.assertRaises(ValueError):
        profiler_impl.start_server(port2)
    finally:
      if server is not None:
        profiler_impl.stop_server()

  def test_double_start_trace_fails(self):
    tmpdir = self.create_tempdir("test_double_start_trace_fails").full_path
    try:
      profiler_impl.start_trace(tmpdir)
      with self.assertRaises(RuntimeError):
        profiler_impl.start_trace(tmpdir)
    finally:
      if profiler_impl._profile_state.profile_session is not None:
        profiler_impl.stop_trace()

  def test_stop_trace_without_start_fails(self):
    with self.assertRaises(RuntimeError):
      profiler_impl.stop_trace()


class ProfilerIntegrationErrorsTest(seed_test_utils.RepeatableTest):

  def test_invalid_profiler_config(self):
    device = torch.device("tpu")
    output_dir = self.create_tempdir("invalid_config").full_path
    _cleanup_profile_dir()

    bad_config = torch.profiler._ExperimentalConfig(
        custom_profiler_config="invalid_option_missing_colon"
    )

    with self.assertRaisesRegex(
        RuntimeError,
        r"expected the config item to be in the 'key:value' format, got"
        r" 'invalid_option_missing_colon'",
    ):
      with torch.profiler.profile(
          activities=[
              torch.profiler.ProfilerActivity.CPU,
              torch.profiler.ProfilerActivity.TPU,  # type: ignore
          ],
          on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
          experimental_config=bad_config,
      ):
        _ = torch.ones((16, 16)).to(device) @ torch.ones((16, 16)).to(device)

  @unittest.skip(
      "Skipped until new libtpu wheel containing BrokenProfiler is rolled into"
      " google3 (bootstrap dependency)."
  )
  def test_tpu_profiler_config_invalid_experimental_option(self):
    device = torch.device("tpu")
    output_dir = pathlib.Path(
        self.create_tempdir("invalid_experimental_option").full_path
    )

    config = TpuProfilerConfig(
        host_tracer_level=2,
        experimental_options={
            "invalid_option_key": "some_value",
        },
        run_dir=output_dir,
    )

    with self.assertRaisesRegex(
        RuntimeError,
        r"Parsing advanced_configuration failed\. The following keys "
        r"were not recognized: invalid_option_key",
    ):
      with torch.profiler.profile(
          activities=[
              torch.profiler.ProfilerActivity.CPU,
              torch.profiler.ProfilerActivity.TPU,  # type: ignore
          ],
          on_trace_ready=torch.profiler.tensorboard_trace_handler(
              str(output_dir)
          ),
          experimental_config=config,
      ):
        a = torch.ones((16, 16)).to(device)
        b = torch.ones((16, 16)).to(device)
        _ = a @ b

  @parameterized.named_parameters(
      dict(testcase_name="colon", key="invalid:key", value="value"),
      dict(testcase_name="comma", key="invalid,key", value="value"),
  )
  def test_tpu_profiler_config_invalid_characters_in_experimental_options(
      self, key, value
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"Experimental option keys cannot contain ':' or ','",
    ):
      TpuProfilerConfig(experimental_options={key: value})

  def test_tpu_profiler_config_worker_rank_validation(self):
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=0)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=42)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=-1)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=3.5)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=True)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank={"rank": 1})
    with self.assertRaises(ValueError):
      TpuProfilerConfig(worker_rank="worker:1")
    with self.assertRaises(ValueError):
      TpuProfilerConfig(worker_rank="worker,1")


if __name__ == "__main__":
  absltest.main()
