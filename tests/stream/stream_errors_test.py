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

"""Tests for stream and event error handling in TorchTPU."""

from absl.testing import absltest
import torch
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
from tests import seed_test_utils

_NOT_IMPLEMENTED_MSG = (
    'Streams and Events are not fully implemented in TorchTPU. Please file'
    ' a feature request describing your use case.'
)


class StreamErrorsTest(seed_test_utils.RepeatableTest):

  def test_stream_unimplemented_method(self):
    default_stream = torch.tpu.default_stream()
    with self.assertRaisesWithLiteralMatch(
        NotImplementedError, _NOT_IMPLEMENTED_MSG
    ):
      default_stream.query()

  def test_stream_partially_implemented_method(self):
    s = torch.tpu.Stream()
    with self.assertRaisesWithLiteralMatch(
        NotImplementedError, _NOT_IMPLEMENTED_MSG
    ):
      s.query()
    with self.assertRaisesWithLiteralMatch(
        NotImplementedError, _NOT_IMPLEMENTED_MSG
    ):
      s.priority_range()

  def test_event_record_invalid_type_raises(self):
    event = torch.tpu.Event()
    with self.assertRaisesRegex(
        TypeError, 'expected torch.Stream, but got str'
    ):
      event.record('invalid')

  def test_event_partially_implemented(self):
    e = torch.tpu.Event()
    with self.assertRaisesWithLiteralMatch(
        NotImplementedError, _NOT_IMPLEMENTED_MSG
    ):
      e.ipc_handle()


if __name__ == '__main__':
  absltest.main()
