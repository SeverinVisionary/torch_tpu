/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_FUSE_MANUAL_FLASH_ATTENTION_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_FUSE_MANUAL_FLASH_ATTENTION_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that fuses manual batched-matmul scaled dot-product
// attention patterns (`bmm(Q, K^T) -> [scale] -> [bias] -> softmax -> bmm(P,
// V)`) into the Mosaic FlashAttention kernel when the `f32` attention score
// matrix exceeds TensorCore scoped VMEM capacity.
std::unique_ptr<mlir::Pass> createFuseManualFlashAttentionPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_FUSE_MANUAL_FLASH_ATTENTION_H_
