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

#include "csrc/ops/hardsigmoid/hardsigmoid_aten_kernels.h"

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildHardsigmoidShlo(mlir::MlirOp input_op) {
  auto type = GetTensorTypeOrDie(input_op);
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));

  TT_ASSIGN_OR_RETURN(auto compute_dtype, InferComputationDtype(element_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp x, CastIfNeeded(input_op, compute_dtype));

  // hardsigmoid(x) = clamp(0, x + 3, 6) * (1/6)
  mlir::MlirOp zero = MakeConstantLike(x, 0.0, compute_dtype);
  mlir::MlirOp three = MakeConstantLike(x, 3.0, compute_dtype);
  mlir::MlirOp six = MakeConstantLike(x, 6.0, compute_dtype);
  mlir::MlirOp one_sixth = MakeConstantLike(x, 1.0 / 6.0, compute_dtype);

  mlir::MlirOp x_shifted = mlir::stablehlo::Add(x, three);
  mlir::MlirOp clamped = mlir::stablehlo::Clamp(zero, x_shifted, six);
  mlir::MlirOp result = mlir::stablehlo::Mul(clamped, one_sixth);

  return CastIfNeeded(result, element_type);
}

absl::StatusOr<mlir::MlirOp> BuildHardsigmoidBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self) {
  TT_ASSIGN_OR_RETURN((auto [bcast_g, bcast_x]),
                      ApplyBroadcastIfNeeded(grad_output, self));

  const mlir::ElementType g_dtype = GetElementTypeOrDie(bcast_g);
  const mlir::ElementType x_dtype = GetElementTypeOrDie(bcast_x);

  const at::ScalarType g_scalar = ConvertTo<at::ScalarType>(g_dtype);
  const at::ScalarType x_scalar = ConvertTo<at::ScalarType>(x_dtype);
  const at::ScalarType common_scalar = c10::promoteTypes(g_scalar, x_scalar);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType common_dtype,
                      ConvertTo<mlir::ElementType>(common_scalar));

  TT_ASSIGN_OR_RETURN(const auto compute_dtype,
                      InferComputationDtype(common_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp g, CastIfNeeded(bcast_g, compute_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp x, CastIfNeeded(bcast_x, compute_dtype));

  // grad = 1/6 if -3 < x < 3 else 0
  mlir::MlirOp neg_three = MakeConstantLike(x, -3.0, compute_dtype);
  mlir::MlirOp three = MakeConstantLike(x, 3.0, compute_dtype);

  mlir::MlirOp cond_gt = mlir::stablehlo::Compare(
      x, neg_three, mlir::stablehlo::ComparisonDirection::GT);
  mlir::MlirOp cond_lt = mlir::stablehlo::Compare(
      x, three, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp in_bounds = mlir::stablehlo::And(cond_gt, cond_lt);

  mlir::MlirOp zero = MakeConstantLike(x, 0.0, compute_dtype);
  mlir::MlirOp one_sixth = MakeConstantLike(x, 1.0 / 6.0, compute_dtype);

  mlir::MlirOp mid_grad = mlir::stablehlo::Mul(g, one_sixth);
  mlir::MlirOp result = mlir::stablehlo::Select(in_bounds, mid_grad, zero);

  return CastIfNeeded(result, common_dtype);
}

}  // namespace

at::Tensor& AtenHardsigmoidOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kHardsigmoidOut, _, (self, out), {
    TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(self.scalar_type());
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, BuildHardsigmoidShlo,
                   {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                    .allow_out_dtype_cast = false}));
    return out;
  });
}

at::Tensor& AtenHardsigmoidBackwardGradInput(const at::Tensor& grad_output,
                                             const at::Tensor& self,
                                             at::Tensor& grad_input) {
  TT_KERNEL(OpName::kHardsigmoidBackwardGradInput, _,
            (grad_output, self, grad_input), {
              TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
                  << "expected the input dtype to be floating point, got "
                  << ToString(self.scalar_type());
              TT_THROW_IF_ERROR(BinaryOpOut(
                  grad_output, self, grad_input, BuildHardsigmoidBackwardShlo,
                  {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
              return grad_input;
            });
}

}  // namespace torch_tpu
