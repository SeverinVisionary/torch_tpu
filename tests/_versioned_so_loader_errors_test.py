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

"""Error tests for the version-dispatch selection logic."""

from absl.testing import absltest
from absl.testing import parameterized
from torch_tpu import _versioned_so_loader as loader
from tests import seed_test_utils


class ResolveSuffixErrorsTest(seed_test_utils.RepeatableTest):

  @parameterized.named_parameters(
      ("release_older_than_every_glue", "2_9_0", "predates"),
      ("undetectable_running_version", None, "could not determine"),
  )
  def test_raises_import_error(self, running, message):
    with self.assertRaisesRegex(ImportError, message):
      loader.resolve_suffix(running, {"2_11_0", "2_12_0"})


if __name__ == "__main__":
  absltest.main()
