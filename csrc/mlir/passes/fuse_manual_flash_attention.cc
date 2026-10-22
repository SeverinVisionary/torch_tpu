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

#include "csrc/mlir/passes/fuse_manual_flash_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_kernel.h"
#include "csrc/ops/scaled_dot_product_attention/util.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

// GENERIC: hardware TPU TensorCore vector register lane width.
constexpr int64_t kTpuLaneWidth = 128;

// GENERIC: cost-model Minimum double-buffered Mosaic block size
// (2 * kTpuLaneWidth).
constexpr int64_t kMinHeuristicTileSize = 2 * kTpuLaneWidth;

// GENERIC: cost-model Target KV compute sub-tile size (8 * kTpuLaneWidth)
// bounding live [qt, kc] softmax intermediates in VMEM.
constexpr int64_t kTargetKvComputeTileSize = 8 * kTpuLaneWidth;

// GENERIC: hardware Default 16 MiB TensorCore scoped VMEM allocation limit.
constexpr int64_t kTpuScopedVmemLimitBytes = 16 * 1024 * 1024;

int64_t RoundUpToTileSize(int64_t seq_len, int64_t tile_size) {
  return llvm::divideCeil(seq_len, tile_size) * tile_size;
}

int64_t GetInitialTileSize(int64_t seq_len) {
  if (seq_len <= kTpuLaneWidth) {
    return kTpuLaneWidth;
  }
  if (seq_len <= kMinHeuristicTileSize) {
    return kMinHeuristicTileSize;
  }
  return RoundUpToTileSize(seq_len, kMinHeuristicTileSize);
}

int64_t EstimateTileVmemBytes(int64_t qt, int64_t kt, int64_t qk_head_dim,
                              int64_t vo_head_dim, bool is_f32, bool has_mask) {
  // GENERIC: cost-model Bytes per element (4 for f32, 2 for bf16).
  const int64_t elem_size = is_f32 ? 4 : 2;
  // GENERIC: cost-model Double-buffered Q/K/V/O VMEM tiles plus scratch row
  // statistics and 2 live f32 [qt, kt] tiles (S, P) plus optional mask tile.
  const int64_t double_buffered_bytes =
      2 * (qt * (qk_head_dim * elem_size + vo_head_dim * (elem_size + 4)) +
           kt * (qk_head_dim + vo_head_dim) * elem_size);
  const int64_t scratch_stats_bytes = 1024 * qt;
  const int64_t logits_bytes = 4 * qt * kt;
  const int64_t mask_bytes = has_mask ? elem_size * qt * kt : 0;
  return double_buffered_bytes + scratch_stats_bytes + 2 * logits_bytes +
         mask_bytes;
}

int64_t GetKvComputeTileSize(int64_t kt) {
  int64_t target_kc = kTargetKvComputeTileSize;
  if (kt < target_kc) {
    return kt;
  }
  while (target_kc >= kMinHeuristicTileSize) {
    if (kt % target_kc == 0) {
      return target_kc;
    }
    target_kc -= kMinHeuristicTileSize;
  }
  return kTpuLaneWidth;
}

Tiling ComputeFlashTiling(int64_t q_seq_len, int64_t k_seq_len,
                          int64_t qk_head_dim, int64_t vo_head_dim, bool is_f32,
                          bool has_mask) {
  int64_t qt = GetInitialTileSize(q_seq_len);
  int64_t kt = GetInitialTileSize(k_seq_len);

  while (EstimateTileVmemBytes(qt, kt, qk_head_dim, vo_head_dim, is_f32,
                               has_mask) > kTpuScopedVmemLimitBytes &&
         (qt > kMinHeuristicTileSize || kt > kMinHeuristicTileSize)) {
    if (qt > kt && qt > kMinHeuristicTileSize) {
      qt = std::max(kMinHeuristicTileSize,
                    RoundUpToTileSize(qt / 2, kMinHeuristicTileSize));
    } else if (kt > kMinHeuristicTileSize) {
      kt = std::max(kMinHeuristicTileSize,
                    RoundUpToTileSize(kt / 2, kMinHeuristicTileSize));
    } else if (qt > kMinHeuristicTileSize) {
      qt = std::max(kMinHeuristicTileSize,
                    RoundUpToTileSize(qt / 2, kMinHeuristicTileSize));
    }
  }

  return {
      .qt = qt,
      .kt = kt,
      .kc = GetKvComputeTileSize(kt),
  };
}

bool IsIdentityBroadcastDims(llvm::ArrayRef<int64_t> dims, int64_t rank) {
  if (static_cast<int64_t>(dims.size()) != rank) {
    return false;
  }
  for (int64_t i = 0; i < rank; ++i) {
    if (dims[i] != i) {
      return false;
    }
  }
  return true;
}

bool IsDropLastDimBroadcastDims(llvm::ArrayRef<int64_t> dims, int64_t rank) {
  if (static_cast<int64_t>(dims.size()) != rank - 1) {
    return false;
  }
  for (int64_t i = 0; i < rank - 1; ++i) {
    if (dims[i] != i) {
      return false;
    }
  }
  return true;
}

bool IsBinaryReduceBody(mlir::Region& region, bool is_max) {
  if (region.empty()) {
    return false;
  }
  mlir::Block& block = region.front();
  if (block.getNumArguments() != 2) {
    return false;
  }
  auto ret_op =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  if (!ret_op || ret_op.getOperands().size() != 1) {
    return false;
  }
  mlir::Operation* body_op = ret_op.getOperands()[0].getDefiningOp();
  if (!body_op || body_op->getNumOperands() != 2) {
    return false;
  }
  const bool op_matches = is_max ? mlir::isa<mlir::stablehlo::MaxOp>(body_op)
                                 : mlir::isa<mlir::stablehlo::AddOp>(body_op);
  if (!op_matches) {
    return false;
  }
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg1 = block.getArgument(1);
  mlir::Value lhs = body_op->getOperand(0);
  mlir::Value rhs = body_op->getOperand(1);
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

mlir::Value StripReshapeAndIdentityBroadcasts(mlir::Value cur) {
  while (cur) {
    if (auto reshape = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      auto in_type = mlir::dyn_cast<mlir::RankedTensorType>(
          reshape.getOperand().getType());
      auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(reshape.getType());
      if (!in_type || !out_type || !in_type.hasStaticShape() ||
          !out_type.hasStaticShape() ||
          in_type.getNumElements() != out_type.getNumElements()) {
        break;
      }
      cur = reshape.getOperand();
      continue;
    }
    if (auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      auto in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(bcast.getOperand().getType());
      auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast.getType());
      if (in_type && out_type && in_type == out_type &&
          IsIdentityBroadcastDims(bcast.getBroadcastDimensions(),
                                  out_type.getRank())) {
        cur = bcast.getOperand();
        continue;
      }
    }
    break;
  }
  return cur;
}

mlir::Value UnwrapBroadcastedLastDimValue(mlir::Value val, int64_t rank) {
  auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast) {
    return nullptr;
  }
  llvm::ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
  if (!IsDropLastDimBroadcastDims(dims, rank) &&
      !IsIdentityBroadcastDims(dims, rank)) {
    return nullptr;
  }
  return StripReshapeAndIdentityBroadcasts(bcast.getOperand());
}

bool ExtractSplatFloatConstant(mlir::Value val, float& out_val) {
  mlir::Value cur = val;
  while (cur) {
    if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
      cur = conv.getOperand();
      continue;
    }
    if (auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      if (bcast.getBroadcastDimensions().empty()) {
        cur = bcast.getOperand();
        continue;
      }
    }
    break;
  }
  auto const_op = cur.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense || !dense.isSplat() ||
      !mlir::isa<mlir::FloatType>(dense.getElementType())) {
    return false;
  }
  out_val = static_cast<float>(
      dense.getSplatValue<llvm::APFloat>().convertToDouble());
  return true;
}

bool MatchSingleDimReduce(mlir::Value val, int64_t dim, bool is_max,
                          mlir::Value& reduce_input,
                          mlir::stablehlo::ReduceOp& out_reduce_op) {
  auto reduce_op = val.getDefiningOp<mlir::stablehlo::ReduceOp>();
  if (!reduce_op || reduce_op.getNumResults() != 1 ||
      reduce_op.getInputs().size() != 1 ||
      reduce_op.getInitValues().size() != 1) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
  if (dims.size() != 1 || dims[0] != dim ||
      !IsBinaryReduceBody(reduce_op.getBody(), is_max)) {
    return false;
  }
  reduce_input = reduce_op.getInputs()[0];
  out_reduce_op = reduce_op;
  return true;
}

bool HasExactlyTwoUses(mlir::Value val, mlir::Operation* user_a,
                       mlir::Operation* user_b) {
  int count = 0;
  for (mlir::Operation* user : val.getUsers()) {
    if (user != user_a && user != user_b) {
      return false;
    }
    ++count;
  }
  return count == 2;
}

bool MatchSoftmaxScores(mlir::Value probs_3d, int64_t batch_heads,
                        int64_t q_len, int64_t kv_len,
                        mlir::Value& out_scores) {
  mlir::Value cur = probs_3d;
  if (auto reshape = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    if (!cur.hasOneUse()) {
      return false;
    }
    auto r_type =
        mlir::dyn_cast<mlir::RankedTensorType>(reshape.getOperand().getType());
    if (!r_type || !r_type.hasStaticShape() || r_type.getRank() != 4 ||
        r_type.getDimSize(0) * r_type.getDimSize(1) != batch_heads ||
        r_type.getDimSize(2) != q_len || r_type.getDimSize(3) != kv_len) {
      return false;
    }
    cur = reshape.getOperand();
  }
  if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    if (!cur.hasOneUse()) {
      return false;
    }
    cur = conv.getOperand();
  }
  if (!cur.hasOneUse()) {
    return false;
  }
  auto probs_type = mlir::dyn_cast<mlir::RankedTensorType>(cur.getType());
  if (!probs_type || !probs_type.hasStaticShape() ||
      !probs_type.getElementType().isF32()) {
    return false;
  }
  const int64_t rank = probs_type.getRank();
  if (rank != 3 && rank != 4) {
    return false;
  }
  const int64_t last_dim = rank - 1;

  mlir::Value exp_val;
  mlir::stablehlo::ReduceOp sum_reduce_op;
  mlir::Operation* norm_op = cur.getDefiningOp();
  if (auto div_op = mlir::dyn_cast_or_null<mlir::stablehlo::DivOp>(norm_op)) {
    exp_val = div_op.getLhs();
    mlir::Value unwrapped =
        UnwrapBroadcastedLastDimValue(div_op.getRhs(), rank);
    mlir::Value sum_in;
    if (!unwrapped ||
        !MatchSingleDimReduce(unwrapped, last_dim, /*is_max=*/false, sum_in,
                              sum_reduce_op) ||
        sum_in != exp_val) {
      return false;
    }
  } else if (auto mul_op =
                 mlir::dyn_cast_or_null<mlir::stablehlo::MulOp>(norm_op)) {
    mlir::Value lhs = mul_op.getLhs();
    mlir::Value rhs = mul_op.getRhs();
    if (!lhs.getDefiningOp<mlir::stablehlo::ExpOp>() &&
        rhs.getDefiningOp<mlir::stablehlo::ExpOp>()) {
      std::swap(lhs, rhs);
    }
    exp_val = lhs;
    mlir::Value unwrapped = UnwrapBroadcastedLastDimValue(rhs, rank);
    auto inv_div =
        unwrapped ? unwrapped.getDefiningOp<mlir::stablehlo::DivOp>() : nullptr;
    float one_val = 0.0f;
    if (!inv_div || !ExtractSplatFloatConstant(inv_div.getLhs(), one_val) ||
        one_val != 1.0f) {
      return false;
    }
    mlir::Value denom = StripReshapeAndIdentityBroadcasts(inv_div.getRhs());
    mlir::Value sum_in;
    if (!denom ||
        !MatchSingleDimReduce(denom, last_dim, /*is_max=*/false, sum_in,
                              sum_reduce_op) ||
        sum_in != exp_val) {
      return false;
    }
  } else {
    return false;
  }

  auto exp_op = exp_val.getDefiningOp<mlir::stablehlo::ExpOp>();
  if (!exp_op ||
      !HasExactlyTwoUses(exp_val, norm_op, sum_reduce_op.getOperation())) {
    return false;
  }
  auto sub_op =
      exp_op.getOperand().getDefiningOp<mlir::stablehlo::SubtractOp>();
  if (!sub_op || !exp_op.getOperand().hasOneUse()) {
    return false;
  }
  mlir::Value scores_f32 = sub_op.getLhs();
  mlir::Value unwrapped_max =
      UnwrapBroadcastedLastDimValue(sub_op.getRhs(), rank);
  mlir::Value max_in;
  mlir::stablehlo::ReduceOp max_reduce_op;
  if (!unwrapped_max ||
      !MatchSingleDimReduce(unwrapped_max, last_dim, /*is_max=*/true, max_in,
                            max_reduce_op) ||
      max_in != scores_f32 ||
      !HasExactlyTwoUses(scores_f32, sub_op.getOperation(),
                         max_reduce_op.getOperation())) {
    return false;
  }

  out_scores = scores_f32;
  if (auto conv = out_scores.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    auto src_type =
        mlir::dyn_cast<mlir::RankedTensorType>(conv.getOperand().getType());
    if (src_type &&
        (src_type.getElementType().isBF16() ||
         src_type.getElementType().isF32()) &&
        conv.getOperand().hasOneUse()) {
      out_scores = conv.getOperand();
    }
  }
  return true;
}

mlir::Value StripUnitBatchReshape(mlir::Value val, int64_t batch_heads,
                                  int64_t q_len, int64_t kv_len) {
  auto reshape = val.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshape || !val.hasOneUse()) {
    return val;
  }
  auto in_type =
      mlir::dyn_cast<mlir::RankedTensorType>(reshape.getOperand().getType());
  if (in_type && in_type.hasStaticShape() && in_type.getRank() == 3 &&
      in_type.getDimSize(0) == batch_heads && in_type.getDimSize(1) == q_len &&
      in_type.getDimSize(2) == kv_len) {
    return reshape.getOperand();
  }
  return val;
}

bool MatchScaledBmm1Chain(mlir::Value val, int64_t batch_heads, int64_t q_len,
                          int64_t kv_len, mlir::Type expected_elem_type,
                          mlir::stablehlo::DotGeneralOp& out_bmm1,
                          float& out_scale) {
  auto has_valid_uses = [&](mlir::Value v) {
    return v.hasOneUse() || (v == val && llvm::hasNItems(v.getUsers(), 2));
  };

  mlir::Value cur = StripUnitBatchReshape(val, batch_heads, q_len, kv_len);
  float scale = 1.0f;

  if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    if (has_valid_uses(cur)) {
      cur =
          StripUnitBatchReshape(conv.getOperand(), batch_heads, q_len, kv_len);
    }
  }

  if (auto mul_op = cur.getDefiningOp<mlir::stablehlo::MulOp>()) {
    if (has_valid_uses(cur)) {
      float c = 1.0f;
      if (ExtractSplatFloatConstant(mul_op.getRhs(), c) && c > 0.0f) {
        scale = c;
        cur =
            StripUnitBatchReshape(mul_op.getLhs(), batch_heads, q_len, kv_len);
      } else if (ExtractSplatFloatConstant(mul_op.getLhs(), c) && c > 0.0f) {
        scale = c;
        cur =
            StripUnitBatchReshape(mul_op.getRhs(), batch_heads, q_len, kv_len);
      }
    }
  } else if (auto div_op = cur.getDefiningOp<mlir::stablehlo::DivOp>()) {
    if (has_valid_uses(cur)) {
      float c = 1.0f;
      if (ExtractSplatFloatConstant(div_op.getRhs(), c) && c > 0.0f) {
        scale = 1.0f / c;
        cur =
            StripUnitBatchReshape(div_op.getLhs(), batch_heads, q_len, kv_len);
      }
    }
  }

  if (auto conv = cur.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    if (has_valid_uses(cur)) {
      cur =
          StripUnitBatchReshape(conv.getOperand(), batch_heads, q_len, kv_len);
    }
  }

  auto bmm1 = cur.getDefiningOp<mlir::stablehlo::DotGeneralOp>();
  if (!bmm1 || !has_valid_uses(cur)) {
    return false;
  }
  auto bmm1_type = mlir::dyn_cast<mlir::RankedTensorType>(bmm1.getType());
  if (!bmm1_type || !bmm1_type.hasStaticShape() || bmm1_type.getRank() != 3 ||
      bmm1_type.getElementType() != expected_elem_type ||
      bmm1_type.getDimSize(0) != batch_heads ||
      bmm1_type.getDimSize(1) != q_len || bmm1_type.getDimSize(2) != kv_len) {
    return false;
  }
  mlir::stablehlo::DotDimensionNumbersAttr dims = bmm1.getDotDimensionNumbers();
  if (dims.getLhsBatchingDimensions() != llvm::ArrayRef<int64_t>{0} ||
      dims.getRhsBatchingDimensions() != llvm::ArrayRef<int64_t>{0} ||
      dims.getLhsContractingDimensions() != llvm::ArrayRef<int64_t>{2}) {
    return false;
  }
  if (dims.getRhsContractingDimensions() != llvm::ArrayRef<int64_t>{1} &&
      dims.getRhsContractingDimensions() != llvm::ArrayRef<int64_t>{2}) {
    return false;
  }
  out_bmm1 = bmm1;
  out_scale = scale;
  return true;
}

mlir::Value UnwrapAttentionBiasTo4d(mlir::OpBuilder& builder,
                                    mlir::Location loc, mlir::Value bias,
                                    int64_t batch_heads, int64_t q_len,
                                    int64_t kv_len,
                                    mlir::Type expected_elem_type) {
  mlir::Value cur = bias;
  if (auto bcast = cur.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    if (IsIdentityBroadcastDims(bcast.getBroadcastDimensions(), 4)) {
      auto bcast_in = StripReshapeAndIdentityBroadcasts(bcast.getOperand());
      auto in_type = mlir::dyn_cast<mlir::RankedTensorType>(bcast_in.getType());
      if (in_type && in_type.hasStaticShape() && in_type.getRank() == 4 &&
          in_type.getElementType() == expected_elem_type &&
          in_type.getDimSize(0) == 1 &&
          (in_type.getDimSize(1) == 1 ||
           in_type.getDimSize(1) == batch_heads) &&
          (in_type.getDimSize(2) == 1 || in_type.getDimSize(2) == q_len) &&
          (in_type.getDimSize(3) == 1 || in_type.getDimSize(3) == kv_len)) {
        return bcast_in;
      }
    }
  }
  auto bias_type = mlir::dyn_cast<mlir::RankedTensorType>(cur.getType());
  if (!bias_type || !bias_type.hasStaticShape()) {
    return nullptr;
  }
  if (bias_type.getElementType() != expected_elem_type) {
    if (!mlir::isa<mlir::FloatType>(bias_type.getElementType())) {
      return nullptr;
    }
    bias_type =
        mlir::RankedTensorType::get(bias_type.getShape(), expected_elem_type);
    cur = builder.create<mlir::stablehlo::ConvertOp>(loc, bias_type, cur);
  }
  if (bias_type.getRank() == 3 && bias_type.getDimSize(0) == batch_heads &&
      (bias_type.getDimSize(1) == 1 || bias_type.getDimSize(1) == q_len) &&
      (bias_type.getDimSize(2) == 1 || bias_type.getDimSize(2) == kv_len)) {
    auto type_4d = mlir::RankedTensorType::get(
        {1, batch_heads, bias_type.getDimSize(1), bias_type.getDimSize(2)},
        expected_elem_type);
    return builder.create<mlir::stablehlo::ReshapeOp>(loc, type_4d, cur);
  }
  if (bias_type.getRank() == 4 && bias_type.getDimSize(0) == 1 &&
      (bias_type.getDimSize(1) == 1 ||
       bias_type.getDimSize(1) == batch_heads) &&
      (bias_type.getDimSize(2) == 1 || bias_type.getDimSize(2) == q_len) &&
      (bias_type.getDimSize(3) == 1 || bias_type.getDimSize(3) == kv_len)) {
    return cur;
  }
  return nullptr;
}

mlir::Value Build4dFrom3d(mlir::OpBuilder& builder, mlir::Location loc,
                          mlir::Value val_3d) {
  auto type_3d = mlir::cast<mlir::RankedTensorType>(val_3d.getType());
  auto expected_4d = mlir::RankedTensorType::get(
      {1, type_3d.getDimSize(0), type_3d.getDimSize(1), type_3d.getDimSize(2)},
      type_3d.getElementType());
  if (auto reshape = val_3d.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    if (reshape.getOperand().getType() == expected_4d) {
      return reshape.getOperand();
    }
  }
  return builder.create<mlir::stablehlo::ReshapeOp>(loc, expected_4d, val_3d);
}

mlir::Value BuildKey4d(mlir::OpBuilder& builder, mlir::Location loc,
                       mlir::stablehlo::DotGeneralOp bmm1, int64_t batch_heads,
                       int64_t kv_len, int64_t head_dim) {
  mlir::Value rhs = bmm1.getRhs();
  auto elem_type =
      mlir::cast<mlir::RankedTensorType>(rhs.getType()).getElementType();
  llvm::ArrayRef<int64_t> rhs_contract =
      bmm1.getDotDimensionNumbers().getRhsContractingDimensions();
  if (rhs_contract == llvm::ArrayRef<int64_t>{2}) {
    return Build4dFrom3d(builder, loc, rhs);
  }
  if (auto transpose = rhs.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    llvm::ArrayRef<int64_t> perm = transpose.getPermutation();
    if (perm == llvm::ArrayRef<int64_t>{0, 2, 1}) {
      return Build4dFrom3d(builder, loc, transpose.getOperand());
    }
    if (perm == llvm::ArrayRef<int64_t>{1, 2, 0}) {
      auto t3d_type = mlir::RankedTensorType::get(
          {batch_heads, kv_len, head_dim}, elem_type);
      mlir::Value k_3d = builder.create<mlir::stablehlo::TransposeOp>(
          loc, t3d_type, transpose.getOperand(),
          builder.getDenseI64ArrayAttr({1, 0, 2}));
      return Build4dFrom3d(builder, loc, k_3d);
    }
  }
  auto t3d_type =
      mlir::RankedTensorType::get({batch_heads, kv_len, head_dim}, elem_type);
  mlir::Value k_3d = builder.create<mlir::stablehlo::TransposeOp>(
      loc, t3d_type, rhs, builder.getDenseI64ArrayAttr({0, 2, 1}));
  return Build4dFrom3d(builder, loc, k_3d);
}

mlir::Value PadSequenceDim4d(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value input, int64_t seq_dim,
                             int64_t target_len) {
  auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
  const int64_t current_len = input_type.getDimSize(seq_dim);
  if (current_len == target_len) {
    return input;
  }
  mlir::Type elem_type = input_type.getElementType();
  auto scalar_type = mlir::RankedTensorType::get({}, elem_type);
  auto zero_attr = mlir::DenseElementsAttr::get(
      scalar_type, builder.getFloatAttr(elem_type, 0.0));
  mlir::Value zero =
      builder.create<mlir::stablehlo::ConstantOp>(loc, scalar_type, zero_attr);
  llvm::SmallVector<int64_t, 4> low(4, 0);
  llvm::SmallVector<int64_t, 4> high(4, 0);
  llvm::SmallVector<int64_t, 4> interior(4, 0);
  high[seq_dim] = target_len - current_len;
  llvm::SmallVector<int64_t, 4> out_shape(input_type.getShape());
  out_shape[seq_dim] = target_len;
  auto out_type = mlir::RankedTensorType::get(out_shape, elem_type);
  return builder.create<mlir::stablehlo::PadOp>(
      loc, out_type, input, zero, builder.getDenseI64ArrayAttr(low),
      builder.getDenseI64ArrayAttr(high),
      builder.getDenseI64ArrayAttr(interior));
}

mlir::Value SliceSequenceDim4d(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value input, int64_t seq_dim,
                               int64_t target_len) {
  auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
  const int64_t current_len = input_type.getDimSize(seq_dim);
  if (current_len == target_len) {
    return input;
  }
  llvm::SmallVector<int64_t, 4> start(4, 0);
  llvm::SmallVector<int64_t, 4> limit(input_type.getShape());
  limit[seq_dim] = target_len;
  llvm::SmallVector<int64_t, 4> strides(4, 1);
  auto out_type =
      mlir::RankedTensorType::get(limit, input_type.getElementType());
  return builder.create<mlir::stablehlo::SliceOp>(
      loc, out_type, input, builder.getDenseI64ArrayAttr(start),
      builder.getDenseI64ArrayAttr(limit),
      builder.getDenseI64ArrayAttr(strides));
}

bool TryFuseManualFlashAttention(mlir::func::FuncOp func,
                                 mlir::stablehlo::DotGeneralOp bmm2) {
  auto bmm2_type = mlir::dyn_cast<mlir::RankedTensorType>(bmm2.getType());
  if (!bmm2_type || !bmm2_type.hasStaticShape() || bmm2_type.getRank() != 3) {
    return false;
  }
  mlir::Type elem_type = bmm2_type.getElementType();
  if (!elem_type.isBF16() && !elem_type.isF32()) {
    return false;
  }
  mlir::stablehlo::DotDimensionNumbersAttr bmm2_dims =
      bmm2.getDotDimensionNumbers();
  if (bmm2_dims.getLhsBatchingDimensions() != llvm::ArrayRef<int64_t>{0} ||
      bmm2_dims.getRhsBatchingDimensions() != llvm::ArrayRef<int64_t>{0} ||
      bmm2_dims.getLhsContractingDimensions() != llvm::ArrayRef<int64_t>{2} ||
      bmm2_dims.getRhsContractingDimensions() != llvm::ArrayRef<int64_t>{1}) {
    return false;
  }

  auto probs_type =
      mlir::dyn_cast<mlir::RankedTensorType>(bmm2.getLhs().getType());
  auto v_type = mlir::dyn_cast<mlir::RankedTensorType>(bmm2.getRhs().getType());
  if (!probs_type || !v_type || !probs_type.hasStaticShape() ||
      !v_type.hasStaticShape() || probs_type.getRank() != 3 ||
      v_type.getRank() != 3 || v_type.getElementType() != elem_type) {
    return false;
  }

  const int64_t batch_heads = probs_type.getDimSize(0);
  const int64_t q_len = probs_type.getDimSize(1);
  const int64_t kv_len = probs_type.getDimSize(2);
  const int64_t vo_head_dim = v_type.getDimSize(2);

  // GENERIC: hardware Mosaic FlashAttention 2D sequence tiling requires both
  // query and key sequence dimensions to exceed a single 128-lane tile.
  if (batch_heads <= 0 || q_len <= kTpuLaneWidth || kv_len <= kTpuLaneWidth) {
    return false;
  }
  // GENERIC: cost-model Only fuse manual attention when the materialized f32
  // attention score tensor [B*H, S_q, S_kv] exceeds TensorCore scoped VMEM
  // capacity (16 MiB), forcing HBM spills between QK^T, softmax, and PV.
  const int64_t f32_score_bytes =
      batch_heads * q_len * kv_len * static_cast<int64_t>(sizeof(float));
  if (f32_score_bytes <= kTpuScopedVmemLimitBytes) {
    return false;
  }

  mlir::Value scores;
  if (!MatchSoftmaxScores(bmm2.getLhs(), batch_heads, q_len, kv_len, scores)) {
    return false;
  }

  mlir::stablehlo::DotGeneralOp bmm1;
  float scale = 1.0f;
  mlir::Value raw_bias;
  if (auto add_op = scores.getDefiningOp<mlir::stablehlo::AddOp>()) {
    if (!scores.hasOneUse() && !llvm::hasNItems(scores.getUsers(), 2)) {
      return false;
    }
    if (MatchScaledBmm1Chain(add_op.getLhs(), batch_heads, q_len, kv_len,
                             elem_type, bmm1, scale)) {
      raw_bias = add_op.getRhs();
    } else if (MatchScaledBmm1Chain(add_op.getRhs(), batch_heads, q_len, kv_len,
                                    elem_type, bmm1, scale)) {
      raw_bias = add_op.getLhs();
    } else {
      return false;
    }
  } else if (!MatchScaledBmm1Chain(scores, batch_heads, q_len, kv_len,
                                   elem_type, bmm1, scale)) {
    return false;
  }

  auto q_type = mlir::dyn_cast<mlir::RankedTensorType>(bmm1.getLhs().getType());
  if (!q_type || !q_type.hasStaticShape() || q_type.getRank() != 3 ||
      q_type.getElementType() != elem_type) {
    return false;
  }
  const int64_t qk_head_dim = q_type.getDimSize(2);
  if (qk_head_dim <= 0 || vo_head_dim <= 0) {
    return false;
  }

  mlir::OpBuilder builder(bmm2);
  mlir::Location loc = bmm2.getLoc();

  mlir::Value bias_4d;
  if (raw_bias) {
    bias_4d = UnwrapAttentionBiasTo4d(builder, loc, raw_bias, batch_heads,
                                      q_len, kv_len, elem_type);
    if (!bias_4d) {
      return false;
    }
  }

  const bool is_f32 = elem_type.isF32();
  const bool has_bias = static_cast<bool>(bias_4d);
  const Tiling tiling = ComputeFlashTiling(q_len, kv_len, qk_head_dim,
                                           vo_head_dim, is_f32, has_bias);
  if (EstimateTileVmemBytes(tiling.qt, tiling.kt, qk_head_dim, vo_head_dim,
                            is_f32, has_bias) > kTpuScopedVmemLimitBytes) {
    return false;
  }

  FlashAttnConfig config;
  config.element_type = is_f32 ? kiln::f32 : kiln::bf16;
  config.batch_size = 1;
  config.num_heads = batch_heads;
  config.kv_num_heads = batch_heads;
  config.qk_head_dim = qk_head_dim;
  config.vo_head_dim = vo_head_dim;
  config.q_sequence_length = q_len;
  config.kv_sequence_length = kv_len;
  config.padded_q_sequence_length = RoundUpToTileSize(q_len, tiling.qt);
  config.padded_kv_sequence_length = RoundUpToTileSize(kv_len, tiling.kt);
  config.is_causal = false;
  config.scale = scale;
  config.return_lse = false;
  config.has_attn_bias = has_bias;
  config.dropout_rate = 0.0f;

  if (has_bias) {
    auto bias_type = mlir::cast<mlir::RankedTensorType>(bias_4d.getType());
    for (int64_t idx = 0; idx < 4; ++idx) {
      if (bias_type.getDimSize(idx) == 1) {
        config.mask_broadcast_dims.push_back(idx);
      }
    }
  }

  auto kernel_or = CreateKernel(config, tiling);
  if (!kernel_or.ok()) {
    return false;
  }
  auto serialized_kernel_or = std::move(*kernel_or).Serialize();
  if (!serialized_kernel_or.ok()) {
    return false;
  }

  mlir::Value q_4d = Build4dFrom3d(builder, loc, bmm1.getLhs());
  mlir::Value k_4d =
      BuildKey4d(builder, loc, bmm1, batch_heads, kv_len, qk_head_dim);
  mlir::Value v_4d = Build4dFrom3d(builder, loc, bmm2.getRhs());

  mlir::Value q_pad =
      PadSequenceDim4d(builder, loc, q_4d, 2, config.padded_q_sequence_length);
  mlir::Value k_pad =
      PadSequenceDim4d(builder, loc, k_4d, 2, config.padded_kv_sequence_length);
  mlir::Value v_pad =
      PadSequenceDim4d(builder, loc, v_4d, 2, config.padded_kv_sequence_length);

  llvm::SmallVector<mlir::Value, 4> operands = {q_pad, k_pad, v_pad};
  if (has_bias) {
    auto bias_type = mlir::cast<mlir::RankedTensorType>(bias_4d.getType());
    mlir::Value b_pad = bias_4d;
    if (bias_type.getDimSize(2) != 1) {
      b_pad = PadSequenceDim4d(builder, loc, b_pad, 2,
                               config.padded_q_sequence_length);
    }
    if (bias_type.getDimSize(3) != 1) {
      b_pad = PadSequenceDim4d(builder, loc, b_pad, 3,
                               config.padded_kv_sequence_length);
    }
    operands.push_back(b_pad);
  }

  auto padded_out_type = mlir::RankedTensorType::get(
      {1, batch_heads, config.padded_q_sequence_length, vo_head_dim},
      elem_type);

  std::string backend_config_json = absl::StrFormat(
      R"({
        "custom_call_config": {
          "body": "%s",
          "needs_layout_passes": true,
          "serialization_format": 1
        },
        "device_type": "DEVICE_TYPE_TENSORCORE",
        "scoped_memory_configs": [
          {
            "memory_space": 1,
            "offset": 0,
            "size": %d
          }
        ]
      })",
      absl::Base64Escape(*serialized_kernel_or), kTpuScopedVmemLimitBytes);

  auto layout_type = mlir::RankedTensorType::get({4}, builder.getIndexType());
  auto default_layout = mlir::DenseIntElementsAttr::get(
      layout_type, llvm::ArrayRef<int64_t>{3, 2, 1, 0});
  llvm::SmallVector<mlir::Attribute, 4> input_layouts(operands.size(),
                                                      default_layout);
  llvm::SmallVector<mlir::Attribute, 1> output_layouts = {default_layout};

  auto custom_call = mlir::stablehlo::CustomCallOp::create(
      builder, loc, mlir::TypeRange{padded_out_type}, operands,
      builder.getStringAttr("tpu_custom_call"), builder.getBoolAttr(false),
      builder.getStringAttr(backend_config_json),
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          builder.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING),
      builder.getArrayAttr({}), builder.getArrayAttr(input_layouts),
      builder.getArrayAttr(output_layouts),
      /*output_operand_aliases=*/nullptr,
      /*result_tilings=*/nullptr);

  mlir::Value out_4d =
      SliceSequenceDim4d(builder, loc, custom_call.getResult(0), 2, q_len);
  mlir::Value out_3d =
      builder.create<mlir::stablehlo::ReshapeOp>(loc, bmm2_type, out_4d);
  bmm2.replaceAllUsesWith(out_3d);
  return true;
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&](mlir::Operation* op) {
      if (mlir::isOpTriviallyDead(op)) {
        dead_ops.push_back(op);
      }
    });
    for (auto it = dead_ops.rbegin(); it != dead_ops.rend(); ++it) {
      if (mlir::isOpTriviallyDead(*it)) {
        (*it)->erase();
        changed = true;
      }
    }
  }
}

struct FuseManualFlashAttentionPass
    : public mlir::PassWrapper<FuseManualFlashAttentionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseManualFlashAttentionPass)

  llvm::StringRef getArgument() const override {
    return "fuse-manual-flash-attention";
  }
  llvm::StringRef getDescription() const override {
    return "Fuses manual bmm+softmax+bmm attention into Mosaic FlashAttention "
           "when the attention score matrix exceeds scoped VMEM capacity.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::stablehlo::DotGeneralOp> dot_ops;
    func.walk([&](mlir::stablehlo::DotGeneralOp op) { dot_ops.push_back(op); });
    bool any_fused = false;
    for (mlir::stablehlo::DotGeneralOp dot_op : dot_ops) {
      if (TryFuseManualFlashAttention(func, dot_op)) {
        any_fused = true;
      }
    }
    if (any_fused) {
      EraseTriviallyDeadOps(func);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createFuseManualFlashAttentionPass() {
  return std::make_unique<FuseManualFlashAttentionPass>();
}

}  // namespace torch_tpu
