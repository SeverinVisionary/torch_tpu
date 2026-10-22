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

TEST(PipelineTest,
     PropagatesCustomCallOutputLayoutAndCanonicalizesUnitBatchTranspose) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%q: tensor<1x12x512x64xbf16>, %kv_proj: tensor<512x768xbf16>) -> (tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>) {
    %r4d = stablehlo.reshape %kv_proj : (tensor<512x768xbf16>) -> tensor<512x12x64x1xbf16>
    %kv = stablehlo.transpose %r4d, dims = [3, 1, 0, 2] : (tensor<512x12x64x1xbf16>) -> tensor<1x12x512x64xbf16>
    %attn = stablehlo.custom_call @tpu_custom_call(%q, %kv, %kv) {
      api_version = 2 : i32,
      backend_config = "{}",
      has_side_effect = false,
      operand_layouts = [
        dense<[3, 2, 1, 0]> : tensor<4xindex>,
        dense<[3, 2, 1, 0]> : tensor<4xindex>,
        dense<[3, 2, 1, 0]> : tensor<4xindex>
      ],
      result_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>]
    } : (tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>) -> tensor<1x12x512x64xbf16>
    return %attn, %kv : tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>
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

  mlir::func::FuncOp main_fn = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_fn);
  EXPECT_EQ(main_fn.getResultAttr(0, "mhlo.layout_mode"), nullptr);
  auto layout_attr =
      main_fn.getResultAttrOfType<mlir::StringAttr>(1, "mhlo.layout_mode");
  ASSERT_NE(layout_attr, nullptr);
  EXPECT_EQ(layout_attr.getValue(), "{3,2,1,0}");

  mlir::stablehlo::TransposeOp transpose_3d = nullptr;
  int transpose_3d_count = 0;
  int transpose_4d_count = 0;
  module->walk([&](mlir::stablehlo::TransposeOp op) {
    auto rank = mlir::cast<mlir::RankedTensorType>(op.getType()).getRank();
    if (rank == 3) {
      ++transpose_3d_count;
      transpose_3d = op;
    } else if (rank == 4) {
      ++transpose_4d_count;
    }
  });
  EXPECT_EQ(transpose_3d_count, 1);
  EXPECT_EQ(transpose_4d_count, 0);
  ASSERT_NE(transpose_3d, nullptr);
  EXPECT_EQ(transpose_3d.getPermutation(), llvm::ArrayRef<int64_t>({1, 0, 2}));
  auto t3d_type = mlir::cast<mlir::RankedTensorType>(transpose_3d.getType());
  EXPECT_EQ(t3d_type.getShape(), llvm::ArrayRef<int64_t>({12, 512, 64}));

  auto reshape_3d =
      transpose_3d.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
  ASSERT_NE(reshape_3d, nullptr);
  auto r3d_type = mlir::cast<mlir::RankedTensorType>(reshape_3d.getType());
  EXPECT_EQ(r3d_type.getShape(), llvm::ArrayRef<int64_t>({512, 12, 64}));
  EXPECT_EQ(reshape_3d.getOperand(), main_fn.getArgument(1));

  auto return_op = mlir::cast<mlir::func::ReturnOp>(
      main_fn.getBody().front().getTerminator());
  mlir::Value ret1 = return_op.getOperand(1);
  auto reshape_4d = ret1.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  ASSERT_NE(reshape_4d, nullptr);
  EXPECT_EQ(reshape_4d.getOperand(), transpose_3d.getResult());
  auto r4d_type = mlir::cast<mlir::RankedTensorType>(reshape_4d.getType());
  EXPECT_EQ(r4d_type.getShape(), llvm::ArrayRef<int64_t>({1, 12, 512, 64}));

  mlir::stablehlo::CustomCallOp custom_call = nullptr;
  module->walk([&](mlir::stablehlo::CustomCallOp op) { custom_call = op; });
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call.getOperand(1), ret1);
  EXPECT_EQ(custom_call.getOperand(2), ret1);
}

TEST(PipelineTest, PropagatesCustomCallOutputLayoutWhenLseReturnedForBackward) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%q: tensor<1x12x512x64xbf16>, %kv_proj: tensor<512x768xbf16>) -> (tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>) {
    %r4d = stablehlo.reshape %kv_proj : (tensor<512x768xbf16>) -> tensor<512x12x64x1xbf16>
    %kv = stablehlo.transpose %r4d, dims = [3, 1, 0, 2] : (tensor<512x12x64x1xbf16>) -> tensor<1x12x512x64xbf16>
    %attn:2 = stablehlo.custom_call @tpu_custom_call(%q, %kv, %kv) {
      api_version = 2 : i32,
      backend_config = "{}",
      has_side_effect = false,
      operand_layouts = [
        dense<[3, 2, 1, 0]> : tensor<4xindex>,
        dense<[3, 2, 1, 0]> : tensor<4xindex>,
        dense<[3, 2, 1, 0]> : tensor<4xindex>
      ],
      result_layouts = [
        dense<[3, 2, 1, 0]> : tensor<4xindex>,
        dense<[3, 2, 1, 0]> : tensor<4xindex>
      ]
    } : (tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>) -> (tensor<1x12x512x64xbf16>, tensor<1x12x1x512xf32>)
    return %attn#0, %kv : tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>
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

  mlir::func::FuncOp main_fn = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_fn);
  EXPECT_EQ(main_fn.getResultAttr(0, "mhlo.layout_mode"), nullptr);
  auto layout_attr =
      main_fn.getResultAttrOfType<mlir::StringAttr>(1, "mhlo.layout_mode");
  ASSERT_NE(layout_attr, nullptr);
  EXPECT_EQ(layout_attr.getValue(), "{3,2,1,0}");
}

TEST(PipelineTest, PropagatesCustomCallOutputLayoutThroughPadOp) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%q: tensor<1x12x512x64xbf16>, %kv: tensor<1x12x512x64xbf16>) -> (tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>) {
    %c0 = stablehlo.constant dense<0.0> : tensor<bf16>
    %padded = stablehlo.pad %kv, %c0, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x12x512x64xbf16>, tensor<bf16>) -> tensor<1x12x512x64xbf16>
    %attn = stablehlo.custom_call @tpu_custom_call(%q, %padded, %padded) {
      api_version = 2 : i32,
      backend_config = "{}",
      has_side_effect = false,
      operand_layouts = [
        dense<[3, 2, 1, 0]> : tensor<4xindex>,
        dense<[3, 2, 1, 0]> : tensor<4xindex>,
        dense<[3, 2, 1, 0]> : tensor<4xindex>
      ],
      result_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>]
    } : (tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>) -> tensor<1x12x512x64xbf16>
    return %attn, %kv : tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>
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

  mlir::func::FuncOp main_fn = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_fn);
  EXPECT_EQ(main_fn.getResultAttr(0, "mhlo.layout_mode"), nullptr);
  auto layout_attr =
      main_fn.getResultAttrOfType<mlir::StringAttr>(1, "mhlo.layout_mode");
  ASSERT_NE(layout_attr, nullptr);
  EXPECT_EQ(layout_attr.getValue(), "{3,2,1,0}");
}

TEST(PipelineTest,
     DoesNotAbortLayoutPropagationWhenFunctionReturnsBlockArgument) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%passthrough: tensor<1xi32>, %q: tensor<1x12x512x64xbf16>, %kv_proj: tensor<512x768xbf16>) -> (tensor<1xi32>, tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>) {
    %r4d = stablehlo.reshape %kv_proj : (tensor<512x768xbf16>) -> tensor<512x12x64x1xbf16>
    %kv = stablehlo.transpose %r4d, dims = [3, 1, 0, 2] : (tensor<512x12x64x1xbf16>) -> tensor<1x12x512x64xbf16>
    %attn = stablehlo.custom_call @tpu_custom_call(%q, %kv, %kv) {
      api_version = 2 : i32,
      backend_config = "{}",
      has_side_effect = false,
      operand_layouts = [
        dense<[3, 2, 1, 0]> : tensor<4xindex>,
        dense<[3, 2, 1, 0]> : tensor<4xindex>,
        dense<[3, 2, 1, 0]> : tensor<4xindex>
      ],
      result_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>]
    } : (tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>) -> tensor<1x12x512x64xbf16>
    return %passthrough, %attn, %kv : tensor<1xi32>, tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>
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

  mlir::func::FuncOp main_fn = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_fn);
  EXPECT_EQ(main_fn.getResultAttr(0, "mhlo.layout_mode"), nullptr);
  EXPECT_EQ(main_fn.getResultAttr(1, "mhlo.layout_mode"), nullptr);
  auto layout_attr =
      main_fn.getResultAttrOfType<mlir::StringAttr>(2, "mhlo.layout_mode");
  ASSERT_NE(layout_attr, nullptr);
  EXPECT_EQ(layout_attr.getValue(), "{3,2,1,0}");
}

TEST(PipelineTest, PropagatesNonDefaultCustomCallOperandLayout) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%q: tensor<1x12x512x64xbf16>, %kv: tensor<1x12x512x64xbf16>) -> (tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>) {
    %attn = stablehlo.custom_call @tpu_custom_call(%q, %kv, %kv) {
      api_version = 2 : i32,
      backend_config = "{}",
      has_side_effect = false,
      operand_layouts = [
        dense<[0, 1, 2, 3]> : tensor<4xindex>,
        dense<[0, 1, 2, 3]> : tensor<4xindex>,
        dense<[0, 1, 2, 3]> : tensor<4xindex>
      ],
      result_layouts = [dense<[0, 1, 2, 3]> : tensor<4xindex>]
    } : (tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>) -> tensor<1x12x512x64xbf16>
    return %attn, %kv : tensor<1x12x512x64xbf16>, tensor<1x12x512x64xbf16>
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

  mlir::func::FuncOp main_fn = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_fn);
  auto layout_attr =
      main_fn.getResultAttrOfType<mlir::StringAttr>(1, "mhlo.layout_mode");
  ASSERT_NE(layout_attr, nullptr);
  EXPECT_EQ(layout_attr.getValue(), "{0,1,2,3}");
}

TEST(PipelineTest, CanonicalizesUnitBatchTransposeInIntermediateLayer) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x12x512x64xbf16>, %kv_proj: tensor<512x768xbf16>) -> tensor<1x12x512x64xbf16> {
    %r4d = stablehlo.reshape %kv_proj : (tensor<512x768xbf16>) -> tensor<512x12x64x1xbf16>
    %kv = stablehlo.transpose %r4d, dims = [3, 1, 0, 2] : (tensor<512x12x64x1xbf16>) -> tensor<1x12x512x64xbf16>
    %out = stablehlo.add %x, %kv : tensor<1x12x512x64xbf16>
    return %out : tensor<1x12x512x64xbf16>
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

  int transpose_3d_count = 0;
  int transpose_4d_count = 0;
  module->walk([&](mlir::stablehlo::TransposeOp op) {
    auto rank = mlir::cast<mlir::RankedTensorType>(op.getType()).getRank();
    if (rank == 3) {
      ++transpose_3d_count;
    } else if (rank == 4) {
      ++transpose_4d_count;
    }
  });
  EXPECT_EQ(transpose_3d_count, 1);
  EXPECT_EQ(transpose_4d_count, 0);
}

}  // namespace
}  // namespace torch_tpu
