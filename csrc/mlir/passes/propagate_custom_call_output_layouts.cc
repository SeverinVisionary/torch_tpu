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

#include "csrc/mlir/passes/propagate_custom_call_output_layouts.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Returns true if the permutation matches expected.
bool IsExpectedPermutation(llvm::ArrayRef<int64_t> actual,
                           llvm::ArrayRef<int64_t> expected) {
  return actual == expected;
}

// Returns the minor-to-major layout string for the given operand of a custom
// call if specified in operand_layouts, or std::nullopt otherwise.
std::optional<std::string> GetCustomCallOperandLayout(
    mlir::stablehlo::CustomCallOp custom_call, unsigned operand_index) {
  if (custom_call.getCallTargetName() != "tpu_custom_call") {
    return std::nullopt;
  }
  std::optional<mlir::ArrayAttr> layouts = custom_call.getOperandLayouts();
  if (!layouts.has_value() || operand_index >= layouts->size()) {
    return std::nullopt;
  }
  auto dense_attr =
      mlir::dyn_cast<mlir::DenseIntElementsAttr>((*layouts)[operand_index]);
  if (!dense_attr || dense_attr.getNumElements() == 0) {
    return std::nullopt;
  }
  llvm::SmallVector<int64_t, 4> perm;
  for (const llvm::APInt& v : dense_attr.getValues<llvm::APInt>()) {
    perm.push_back(v.getSExtValue());
  }
  return absl::StrCat("{", absl::StrJoin(perm, ","), "}");
}

// Returns true if all uses of pad_op feed custom calls with the same layout.
bool AllPadUsesHaveLayout(mlir::stablehlo::PadOp pad_op,
                          std::string& layout_str, bool& saw_custom_call) {
  for (mlir::OpOperand& use : pad_op->getUses()) {
    auto custom_call =
        mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(use.getOwner());
    if (!custom_call) {
      return false;
    }
    std::optional<std::string> op_layout =
        GetCustomCallOperandLayout(custom_call, use.getOperandNumber());
    if (!op_layout.has_value()) {
      return false;
    }
    if (layout_str.empty()) {
      layout_str = *op_layout;
    } else if (layout_str != *op_layout) {
      return false;
    }
    saw_custom_call = true;
  }
  return true;
}

// Returns the layout string if all non-return uses of val feed custom calls
// (directly or via pad) with the same layout, or std::nullopt otherwise.
std::optional<std::string> GetPropagatedLayout(mlir::Value val,
                                               mlir::func::ReturnOp return_op) {
  std::string layout_str;
  bool saw_custom_call = false;
  for (mlir::OpOperand& use : val.getUses()) {
    mlir::Operation* owner = use.getOwner();
    if (owner == return_op.getOperation()) {
      continue;
    }
    if (auto custom_call =
            mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(owner)) {
      std::optional<std::string> op_layout =
          GetCustomCallOperandLayout(custom_call, use.getOperandNumber());
      if (!op_layout.has_value()) {
        return std::nullopt;
      }
      if (layout_str.empty()) {
        layout_str = *op_layout;
      } else if (layout_str != *op_layout) {
        return std::nullopt;
      }
      saw_custom_call = true;
      continue;
    }
    if (auto pad_op = mlir::dyn_cast<mlir::stablehlo::PadOp>(owner)) {
      if (use.getOperandNumber() != 0 ||
          !AllPadUsesHaveLayout(pad_op, layout_str, saw_custom_call)) {
        return std::nullopt;
      }
      continue;
    }
    return std::nullopt;
  }
  if (!saw_custom_call || layout_str.empty()) {
    return std::nullopt;
  }
  return layout_str;
}

// Returns true if transpose_op is part of a unit-batch 4D transpose chain.
bool MatchUnitBatchTransposeChain(mlir::stablehlo::TransposeOp transpose_op,
                                  mlir::stablehlo::ReshapeOp& reshape_op) {
  auto ret_type =
      mlir::dyn_cast<mlir::RankedTensorType>(transpose_op.getType());
  // GENERIC: semantic Unit-batch 4D `[1, H, S, D]` tensor produced from a 2D
  // projection `[S, H*D]` via `Reshape([S, H, D, 1]) -> Transpose({3,1,0,2})`.
  if (!ret_type || !ret_type.hasStaticShape() || ret_type.getRank() != 4 ||
      ret_type.getDimSize(0) != 1) {
    return false;
  }
  static constexpr int64_t kUnitDimPerm[] = {3, 1, 0, 2};
  if (!IsExpectedPermutation(transpose_op.getPermutation(), kUnitDimPerm)) {
    return false;
  }
  reshape_op =
      transpose_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshape_op) {
    return false;
  }
  auto mid_type = mlir::dyn_cast<mlir::RankedTensorType>(reshape_op.getType());
  auto src_type =
      mlir::dyn_cast<mlir::RankedTensorType>(reshape_op.getOperand().getType());
  // GENERIC: semantic Reshape rank 4 and source rank 2 ensure unit-dim match.
  if (!mid_type || !src_type || !mid_type.hasStaticShape() ||
      !src_type.hasStaticShape() || mid_type.getRank() != 4 ||
      src_type.getRank() != 2 || mid_type.getDimSize(3) != 1) {
    return false;
  }
  return true;
}

// Canonicalizes 4D unit-batch transpose chains into 3D transposes.
void CanonicalizeUnitBatchTransposes(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::TransposeOp> transpose_ops;
  func.walk([&](mlir::stablehlo::TransposeOp transpose_op) {
    mlir::stablehlo::ReshapeOp reshape_op;
    if (MatchUnitBatchTransposeChain(transpose_op, reshape_op)) {
      transpose_ops.push_back(transpose_op);
    }
  });

  for (mlir::stablehlo::TransposeOp transpose_op : transpose_ops) {
    mlir::stablehlo::ReshapeOp reshape_op;
    if (!MatchUnitBatchTransposeChain(transpose_op, reshape_op)) {
      continue;
    }
    mlir::OpBuilder builder(transpose_op);
    mlir::Location loc = transpose_op.getLoc();
    auto ret_type = mlir::cast<mlir::RankedTensorType>(transpose_op.getType());
    const int64_t heads = ret_type.getDimSize(1);
    const int64_t seq_len = ret_type.getDimSize(2);
    const int64_t head_dim = ret_type.getDimSize(3);
    mlir::Type elem_type = ret_type.getElementType();

    auto r3d_type =
        mlir::RankedTensorType::get({seq_len, heads, head_dim}, elem_type);
    auto t3d_type =
        mlir::RankedTensorType::get({heads, seq_len, head_dim}, elem_type);
    mlir::Value r3d = builder.create<mlir::stablehlo::ReshapeOp>(
        loc, r3d_type, reshape_op.getOperand());
    mlir::Value t3d = builder.create<mlir::stablehlo::TransposeOp>(
        loc, t3d_type, r3d, builder.getDenseI64ArrayAttr({1, 0, 2}));
    mlir::Value r4d =
        builder.create<mlir::stablehlo::ReshapeOp>(loc, ret_type, t3d);

    transpose_op.replaceAllUsesWith(r4d);
    transpose_op.erase();
    if (reshape_op->use_empty()) {
      reshape_op.erase();
    }
  }
}

// Propagates custom-call operand layouts to matching return outputs.
void PropagateOutputLayouts(mlir::func::FuncOp func) {
  if (func.getSymName() != "main" || !func.getBody().hasOneBlock()) {
    return;
  }
  auto return_op = mlir::dyn_cast<mlir::func::ReturnOp>(
      func.getBody().front().getTerminator());
  if (!return_op) {
    return;
  }
  mlir::Builder builder(func.getContext());
  for (unsigned i = 0; i < return_op.getNumOperands(); ++i) {
    if (func.getResultAttr(i, "mhlo.layout_mode") != nullptr) {
      continue;
    }
    mlir::Value ret_val = return_op.getOperand(i);
    std::optional<std::string> layout = GetPropagatedLayout(ret_val, return_op);
    if (layout.has_value()) {
      func.setResultAttr(i, "mhlo.layout_mode", builder.getStringAttr(*layout));
    }
  }
}

// Runs canonicalization and layout propagation on func.
void CanonicalizeAndPropagateLayouts(mlir::func::FuncOp func) {
  CanonicalizeUnitBatchTransposes(func);
  PropagateOutputLayouts(func);
}

struct PropagateCustomCallOutputLayoutsPass
    : public mlir::PassWrapper<PropagateCustomCallOutputLayoutsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      PropagateCustomCallOutputLayoutsPass)

  llvm::StringRef getArgument() const override {
    return "torch-tpu-propagate-custom-call-output-layouts";
  }
  llvm::StringRef getDescription() const override {
    return "Canonicalizes unit-batch 4D transpose chains and propagates "
           "custom-call operand layouts to function results.";
  }

  void runOnOperation() override {
    CanonicalizeAndPropagateLayouts(getOperation());
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createPropagateCustomCallOutputLayoutsPass() {
  return std::make_unique<PropagateCustomCallOutputLayoutsPass>();
}

}  // namespace torch_tpu
