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

#include "csrc/mlir/pipeline.h"

#include <cstdint>

#include "absl/status/status_matchers.h"
#include "csrc/common/error_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
using testing::HasSubstr;

TEST(PipelineTest, RunOptimizationPassesFailsOnInvalidModule) {
  mlir::MLIRContext context;
  context
      .loadDialect<mlir::func::FuncDialect>();  // NOLINT(misc-include-cleaner)

  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());

  // Create a non-external FuncOp with an empty block (missing terminator),
  // which fails MLIR post-pass verification during pm.run().
  auto func = mlir::func::FuncOp::create(builder.getUnknownLoc(), "invalid_fn",
                                         builder.getFunctionType({}, {}));
  func.addEntryBlock();
  module->push_back(func);

  EXPECT_THAT(
      RunTorchTpuOptimizationPasses(*module, &context),
      StatusIs(error::kInternal,
               HasSubstr("Failed to run custom MLIR optimization passes")));
}

TEST(PipelineTest, FoldsShiftInvariantPreSubtractionBeforeSoftmax) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%arg0: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %max1 = stablehlo.reduce(%arg0 init: %min_f32) across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %bcast_max1 = stablehlo.broadcast_in_dim %max1, dims = [0] : (tensor<4xf32>) -> tensor<4x16xf32>
    %x = stablehlo.subtract %arg0, %bcast_max1 : tensor<4x16xf32>
    %max2 = stablehlo.reduce(%x init: %min_f32) across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %bcast_max2 = stablehlo.broadcast_in_dim %max2, dims = [0] : (tensor<4xf32>) -> tensor<4x16xf32>
    %shifted = stablehlo.subtract %x, %bcast_max2 : tensor<4x16xf32>
    %exp = stablehlo.exponential %shifted : tensor<4x16xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [1] : (tensor<4x16xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %bcast_sum = stablehlo.broadcast_in_dim %sum, dims = [0] : (tensor<4xf32>) -> tensor<4x16xf32>
    %out = stablehlo.divide %exp, %bcast_sum : tensor<4x16xf32>
    return %out : tensor<4x16xf32>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int max_reduce_count = 0;
  int sub_count = 0;
  module->walk([&](mlir::stablehlo::ReduceOp op) {
    op.getBody().walk([&](mlir::stablehlo::MaxOp) { ++max_reduce_count; });
  });
  module->walk([&](mlir::stablehlo::SubtractOp) { ++sub_count; });

  EXPECT_EQ(max_reduce_count, 1);
  EXPECT_EQ(sub_count, 1);
}

TEST(PipelineTest, FoldsGptOssAttentionSinkConcatSoftmaxSlice) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%attn_weights: tensor<2x4x8x16xbf16>, %sinks: tensor<2x4x8x1xbf16>) -> tensor<2x4x8x16xbf16> {
    %min_bf16 = stablehlo.constant dense<0xFF7F> : tensor<bf16>
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %concat = stablehlo.concatenate %attn_weights, %sinks, dim = 3 : (tensor<2x4x8x16xbf16>, tensor<2x4x8x1xbf16>) -> tensor<2x4x8x17xbf16>
    %max1 = stablehlo.reduce(%concat init: %min_bf16) across dimensions = [3] : (tensor<2x4x8x17xbf16>, tensor<bf16>) -> tensor<2x4x8xbf16>
     reducer(%lhs: tensor<bf16>, %rhs: tensor<bf16>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<bf16>
      stablehlo.return %m : tensor<bf16>
    }
    %max1_keepdim = stablehlo.reshape %max1 : (tensor<2x4x8xbf16>) -> tensor<2x4x8x1xbf16>
    %bcast_max1 = stablehlo.broadcast_in_dim %max1_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xbf16>) -> tensor<2x4x8x17xbf16>
    %centered = stablehlo.subtract %concat, %bcast_max1 : tensor<2x4x8x17xbf16>
    %centered_f32 = stablehlo.convert %centered : (tensor<2x4x8x17xbf16>) -> tensor<2x4x8x17xf32>
    %max2 = stablehlo.reduce(%centered_f32 init: %min_f32) across dimensions = [3] : (tensor<2x4x8x17xf32>, tensor<f32>) -> tensor<2x4x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %max2_keepdim = stablehlo.reshape %max2 : (tensor<2x4x8xf32>) -> tensor<2x4x8x1xf32>
    %bcast_max2 = stablehlo.broadcast_in_dim %max2_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xf32>) -> tensor<2x4x8x17xf32>
    %shifted = stablehlo.subtract %centered_f32, %bcast_max2 : tensor<2x4x8x17xf32>
    %exp = stablehlo.exponential %shifted : tensor<2x4x8x17xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [3] : (tensor<2x4x8x17xf32>, tensor<f32>) -> tensor<2x4x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %sum_keepdim = stablehlo.reshape %sum : (tensor<2x4x8xf32>) -> tensor<2x4x8x1xf32>
    %bcast_sum = stablehlo.broadcast_in_dim %sum_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xf32>) -> tensor<2x4x8x17xf32>
    %probs_f32 = stablehlo.divide %exp, %bcast_sum : tensor<2x4x8x17xf32>
    %probs_bf16 = stablehlo.convert %probs_f32 : (tensor<2x4x8x17xf32>) -> tensor<2x4x8x17xbf16>
    %scores = stablehlo.slice %probs_bf16 [0:2, 0:4, 0:8, 0:16] : (tensor<2x4x8x17xbf16>) -> tensor<2x4x8x16xbf16>
    return %scores : tensor<2x4x8x16xbf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int concat_count = 0;
  int slice_count = 0;
  int dim17_value_count = 0;
  int max_reduce_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ConcatenateOp>(op)) {
      ++concat_count;
    }
    if (mlir::isa<mlir::stablehlo::SliceOp>(op)) {
      ++slice_count;
    }
    for (mlir::Value res : op->getResults()) {
      if (auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(res.getType())) {
        for (int64_t d : rtt.getShape()) {
          if (d == 17) {
            ++dim17_value_count;
          }
        }
      }
    }
    if (auto reduce = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
      reduce.getBody().walk(
          [&](mlir::stablehlo::MaxOp) { ++max_reduce_count; });
    }
  });

  EXPECT_EQ(concat_count, 0);
  EXPECT_EQ(slice_count, 0);
  EXPECT_EQ(dim17_value_count, 0);
  EXPECT_EQ(max_reduce_count, 1);
}

TEST(PipelineTest, FoldsGptOssAttentionSinkConcatSoftmaxSliceMultiTokenSink) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%attn_weights: tensor<2x4x8x16xbf16>, %sinks: tensor<2x4x8x4xbf16>) -> tensor<2x4x8x16xbf16> {
    %min_bf16 = stablehlo.constant dense<0xFF7F> : tensor<bf16>
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %concat = stablehlo.concatenate %attn_weights, %sinks, dim = 3 : (tensor<2x4x8x16xbf16>, tensor<2x4x8x4xbf16>) -> tensor<2x4x8x20xbf16>
    %max1 = stablehlo.reduce(%concat init: %min_bf16) across dimensions = [3] : (tensor<2x4x8x20xbf16>, tensor<bf16>) -> tensor<2x4x8xbf16>
     reducer(%lhs: tensor<bf16>, %rhs: tensor<bf16>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<bf16>
      stablehlo.return %m : tensor<bf16>
    }
    %max1_keepdim = stablehlo.reshape %max1 : (tensor<2x4x8xbf16>) -> tensor<2x4x8x1xbf16>
    %bcast_max1 = stablehlo.broadcast_in_dim %max1_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xbf16>) -> tensor<2x4x8x20xbf16>
    %centered = stablehlo.subtract %concat, %bcast_max1 : tensor<2x4x8x20xbf16>
    %centered_f32 = stablehlo.convert %centered : (tensor<2x4x8x20xbf16>) -> tensor<2x4x8x20xf32>
    %max2 = stablehlo.reduce(%centered_f32 init: %min_f32) across dimensions = [3] : (tensor<2x4x8x20xf32>, tensor<f32>) -> tensor<2x4x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %max2_keepdim = stablehlo.reshape %max2 : (tensor<2x4x8xf32>) -> tensor<2x4x8x1xf32>
    %bcast_max2 = stablehlo.broadcast_in_dim %max2_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xf32>) -> tensor<2x4x8x20xf32>
    %shifted = stablehlo.subtract %centered_f32, %bcast_max2 : tensor<2x4x8x20xf32>
    %exp = stablehlo.exponential %shifted : tensor<2x4x8x20xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [3] : (tensor<2x4x8x20xf32>, tensor<f32>) -> tensor<2x4x8xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %sum_keepdim = stablehlo.reshape %sum : (tensor<2x4x8xf32>) -> tensor<2x4x8x1xf32>
    %bcast_sum = stablehlo.broadcast_in_dim %sum_keepdim, dims = [0, 1, 2, 3] : (tensor<2x4x8x1xf32>) -> tensor<2x4x8x20xf32>
    %probs_f32 = stablehlo.divide %exp, %bcast_sum : tensor<2x4x8x20xf32>
    %probs_bf16 = stablehlo.convert %probs_f32 : (tensor<2x4x8x20xf32>) -> tensor<2x4x8x20xbf16>
    %scores = stablehlo.slice %probs_bf16 [0:2, 0:4, 0:8, 0:16] : (tensor<2x4x8x20xbf16>) -> tensor<2x4x8x16xbf16>
    return %scores : tensor<2x4x8x16xbf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int concat_count = 0;
  int slice_count = 0;
  int dim20_value_count = 0;
  int max_reduce_count = 0;

  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ConcatenateOp>(op)) {
      ++concat_count;
    }
    if (mlir::isa<mlir::stablehlo::SliceOp>(op)) {
      ++slice_count;
    }
    for (mlir::Value res : op->getResults()) {
      if (auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(res.getType())) {
        for (int64_t d : rtt.getShape()) {
          if (d == 20) {
            ++dim20_value_count;
          }
        }
      }
    }
    if (auto reduce = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
      reduce.getBody().walk(
          [&](mlir::stablehlo::MaxOp) { ++max_reduce_count; });
    }
  });

  EXPECT_EQ(concat_count, 0);
  EXPECT_EQ(slice_count, 0);
  EXPECT_EQ(dim20_value_count, 0);
  EXPECT_EQ(max_reduce_count, 2);
}

TEST(PipelineTest, FusesManualFlashAttentionWhenScoreMatrixExceedsScopedVmem) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%q: tensor<16x1022x64xbf16>, %k_raw: tensor<1022x16x64xbf16>, %v: tensor<16x1022x64xbf16>) -> tensor<16x1022x64xbf16> {
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %one_f32 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %kt = stablehlo.transpose %k_raw, dims = [1, 2, 0] : (tensor<1022x16x64xbf16>) -> tensor<16x64x1022xbf16>
    %scores_bf16 = stablehlo.dot_general %q, %kt, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x1022x64xbf16>, tensor<16x64x1022xbf16>) -> tensor<16x1022x1022xbf16>
    %scores = stablehlo.convert %scores_bf16 : (tensor<16x1022x1022xbf16>) -> tensor<16x1022x1022xf32>
    %max = stablehlo.reduce(%scores init: %min_f32) across dimensions = [2] : (tensor<16x1022x1022xf32>, tensor<f32>) -> tensor<16x1022xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %max_3d = stablehlo.reshape %max : (tensor<16x1022xf32>) -> tensor<16x1022x1xf32>
    %max_bcast1 = stablehlo.broadcast_in_dim %max_3d, dims = [0, 1, 2] : (tensor<16x1022x1xf32>) -> tensor<16x1022x1xf32>
    %max_2d = stablehlo.reshape %max_bcast1 : (tensor<16x1022x1xf32>) -> tensor<16x1022xf32>
    %max_bcast2 = stablehlo.broadcast_in_dim %max_2d, dims = [0, 1] : (tensor<16x1022xf32>) -> tensor<16x1022x1022xf32>
    %shifted = stablehlo.subtract %scores, %max_bcast2 : tensor<16x1022x1022xf32>
    %exp = stablehlo.exponential %shifted : tensor<16x1022x1022xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [2] : (tensor<16x1022x1022xf32>, tensor<f32>) -> tensor<16x1022xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %sum_3d = stablehlo.reshape %sum : (tensor<16x1022xf32>) -> tensor<16x1022x1xf32>
    %one_3d = stablehlo.broadcast_in_dim %one_f32, dims = [] : (tensor<f32>) -> tensor<16x1022x1xf32>
    %inv_sum = stablehlo.divide %one_3d, %sum_3d : tensor<16x1022x1xf32>
    %inv_bcast1 = stablehlo.broadcast_in_dim %inv_sum, dims = [0, 1, 2] : (tensor<16x1022x1xf32>) -> tensor<16x1022x1xf32>
    %inv_2d = stablehlo.reshape %inv_bcast1 : (tensor<16x1022x1xf32>) -> tensor<16x1022xf32>
    %inv_bcast2 = stablehlo.broadcast_in_dim %inv_2d, dims = [0, 1] : (tensor<16x1022xf32>) -> tensor<16x1022x1022xf32>
    %probs_f32 = stablehlo.multiply %exp, %inv_bcast2 : tensor<16x1022x1022xf32>
    %probs = stablehlo.convert %probs_f32 : (tensor<16x1022x1022xf32>) -> tensor<16x1022x1022xbf16>
    %out = stablehlo.dot_general %probs, %v, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x1022x1022xbf16>, tensor<16x1022x64xbf16>) -> tensor<16x1022x64xbf16>
    return %out : tensor<16x1022x64xbf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int dot_count = 0;
  int custom_call_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_count;
    }
    if (auto cc = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
      if (cc.getCallTargetName() == "tpu_custom_call") {
        ++custom_call_count;
      }
    }
  });
  EXPECT_EQ(dot_count, 0);
  EXPECT_EQ(custom_call_count, 1);
}

TEST(PipelineTest, Fuses4dScaledMaskedManualFlashAttention) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%q: tensor<12x901x64xbf16>, %k_proj: tensor<901x768xbf16>, %v: tensor<1x12x901x64xbf16>, %mask: tensor<1x1x901x901xbf16>) -> (tensor<12x901x64xbf16>, tensor<1x12x901x64xbf16>) {
    %eight_bf16 = stablehlo.constant dense<8.000000e+00> : tensor<bf16>
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %one_f32 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %k_r4d = stablehlo.reshape %k_proj : (tensor<901x768xbf16>) -> tensor<901x12x64x1xbf16>
    %k_4d = stablehlo.transpose %k_r4d, dims = [3, 1, 0, 2] : (tensor<901x12x64x1xbf16>) -> tensor<1x12x901x64xbf16>
    %k_3d = stablehlo.reshape %k_4d : (tensor<1x12x901x64xbf16>) -> tensor<12x901x64xbf16>
    %kt = stablehlo.transpose %k_3d, dims = [0, 2, 1] : (tensor<12x901x64xbf16>) -> tensor<12x64x901xbf16>
    %bmm1 = stablehlo.dot_general %q, %kt, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x901x64xbf16>, tensor<12x64x901xbf16>) -> tensor<12x901x901xbf16>
    %bmm1_4d = stablehlo.reshape %bmm1 : (tensor<12x901x901xbf16>) -> tensor<1x12x901x901xbf16>
    %eight_4d = stablehlo.broadcast_in_dim %eight_bf16, dims = [] : (tensor<bf16>) -> tensor<1x12x901x901xbf16>
    %scaled = stablehlo.divide %bmm1_4d, %eight_4d : tensor<1x12x901x901xbf16>
    %mask_3d = stablehlo.reshape %mask : (tensor<1x1x901x901xbf16>) -> tensor<1x901x901xbf16>
    %mask_4d = stablehlo.broadcast_in_dim %mask_3d, dims = [0, 2, 3] : (tensor<1x901x901xbf16>) -> tensor<1x12x901x901xbf16>
    %biased = stablehlo.add %scaled, %mask_4d : tensor<1x12x901x901xbf16>
    %scores = stablehlo.convert %biased : (tensor<1x12x901x901xbf16>) -> tensor<1x12x901x901xf32>
    %max = stablehlo.reduce(%scores init: %min_f32) across dimensions = [3] : (tensor<1x12x901x901xf32>, tensor<f32>) -> tensor<1x12x901xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %max_bcast = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2] : (tensor<1x12x901xf32>) -> tensor<1x12x901x901xf32>
    %shifted = stablehlo.subtract %scores, %max_bcast : tensor<1x12x901x901xf32>
    %exp = stablehlo.exponential %shifted : tensor<1x12x901x901xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [3] : (tensor<1x12x901x901xf32>, tensor<f32>) -> tensor<1x12x901xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<1x12x901xf32>) -> tensor<1x12x901x901xf32>
    %probs_f32 = stablehlo.divide %exp, %sum_bcast : tensor<1x12x901x901xf32>
    %probs_4d = stablehlo.convert %probs_f32 : (tensor<1x12x901x901xf32>) -> tensor<1x12x901x901xbf16>
    %probs_3d = stablehlo.reshape %probs_4d : (tensor<1x12x901x901xbf16>) -> tensor<12x901x901xbf16>
    %v_3d = stablehlo.reshape %v : (tensor<1x12x901x64xbf16>) -> tensor<12x901x64xbf16>
    %out = stablehlo.dot_general %probs_3d, %v_3d, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x901x901xbf16>, tensor<12x901x64xbf16>) -> tensor<12x901x64xbf16>
    return %out, %k_4d : tensor<12x901x64xbf16>, tensor<1x12x901x64xbf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int dot_count = 0;
  int custom_call_count = 0;
  int custom_call_operands = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_count;
    }
    if (auto cc = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
      if (cc.getCallTargetName() == "tpu_custom_call") {
        ++custom_call_count;
        custom_call_operands = cc.getNumOperands();
      }
    }
  });
  EXPECT_EQ(dot_count, 0);
  EXPECT_EQ(custom_call_count, 1);
  EXPECT_EQ(custom_call_operands, 4);
}

TEST(PipelineTest, DoesNotFuseManualAttentionWhenScoreMatrixFitsInScopedVmem) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%q: tensor<12x197x64xbf16>, %kt: tensor<12x64x197xbf16>, %v: tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16> {
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %bmm1 = stablehlo.dot_general %q, %kt, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x64xbf16>, tensor<12x64x197xbf16>) -> tensor<12x197x197xbf16>
    %scores = stablehlo.convert %bmm1 : (tensor<12x197x197xbf16>) -> tensor<12x197x197xf32>
    %max = stablehlo.reduce(%scores init: %min_f32) across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %max_bcast = stablehlo.broadcast_in_dim %max, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %shifted = stablehlo.subtract %scores, %max_bcast : tensor<12x197x197xf32>
    %exp = stablehlo.exponential %shifted : tensor<12x197x197xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [2] : (tensor<12x197x197xf32>, tensor<f32>) -> tensor<12x197xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0, 1] : (tensor<12x197xf32>) -> tensor<12x197x197xf32>
    %probs_f32 = stablehlo.divide %exp, %sum_bcast : tensor<12x197x197xf32>
    %probs = stablehlo.convert %probs_f32 : (tensor<12x197x197xf32>) -> tensor<12x197x197xbf16>
    %out = stablehlo.dot_general %probs, %v, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x197x197xbf16>, tensor<12x197x64xbf16>) -> tensor<12x197x64xbf16>
    return %out : tensor<12x197x64xbf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int dot_count = 0;
  int custom_call_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_count;
    }
    if (mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
      ++custom_call_count;
    }
  });
  EXPECT_EQ(dot_count, 2);
  EXPECT_EQ(custom_call_count, 0);
}

TEST(PipelineTest, FusesManualFlashAttentionF32) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%q: tensor<16x1022x64xf32>, %kt: tensor<16x64x1022xf32>, %v: tensor<16x1022x64xf32>) -> tensor<16x1022x64xf32> {
    %min_f32 = stablehlo.constant dense<0xFF7FFFFF> : tensor<f32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %scores = stablehlo.dot_general %q, %kt, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x1022x64xf32>, tensor<16x64x1022xf32>) -> tensor<16x1022x1022xf32>
    %max = stablehlo.reduce(%scores init: %min_f32) across dimensions = [2] : (tensor<16x1022x1022xf32>, tensor<f32>) -> tensor<16x1022xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }
    %max_bcast = stablehlo.broadcast_in_dim %max, dims = [0, 1] : (tensor<16x1022xf32>) -> tensor<16x1022x1022xf32>
    %shifted = stablehlo.subtract %scores, %max_bcast : tensor<16x1022x1022xf32>
    %exp = stablehlo.exponential %shifted : tensor<16x1022x1022xf32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) across dimensions = [2] : (tensor<16x1022x1022xf32>, tensor<f32>) -> tensor<16x1022xf32>
     reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %a = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %a : tensor<f32>
    }
    %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0, 1] : (tensor<16x1022xf32>) -> tensor<16x1022x1022xf32>
    %probs = stablehlo.divide %exp, %sum_bcast : tensor<16x1022x1022xf32>
    %out = stablehlo.dot_general %probs, %v, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<16x1022x1022xf32>, tensor<16x1022x64xf32>) -> tensor<16x1022x64xf32>
    return %out : tensor<16x1022x64xf32>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  ASSERT_THAT(RunTorchTpuOptimizationPasses(*module, &context), IsOk());

  int dot_count = 0;
  int custom_call_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_count;
    }
    if (auto cc = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
      if (cc.getCallTargetName() == "tpu_custom_call") {
        ++custom_call_count;
      }
    }
  });
  EXPECT_EQ(dot_count, 0);
  EXPECT_EQ(custom_call_count, 1);
}

}  // namespace
}  // namespace torch_tpu
