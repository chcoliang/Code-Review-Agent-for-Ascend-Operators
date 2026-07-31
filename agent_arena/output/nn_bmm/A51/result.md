# A51 BatchMatMul Code Review

## Bug: CheckDtypeValid 函数错误拒绝合法的混合精度输入

- **位置**: 第 113 行, `CheckDtypeValid` 函数
- **描述**: 在 `CheckDtypeValid` 函数开头新增了 `if (self->GetDataType() != mat2->GetDataType()) { return false; }` 严格要求 self 和 mat2 的 dtype 必须完全一致。这与 BatchMatMul 的 API 规范冲突——API 支持混合精度输入（如 self=FP16, mat2=FP32），框架会自动进行类型提升（Type Promotion）。此额外检查会拒绝所有合法的混合精度场景。
- **触发输入**:
  ```cpp
  // 在 Ascend 910B 上:
  // self: tensor [2, 3, 4], DT_FLOAT16
  // mat2: tensor [2, 4, 5], DT_FLOAT (合法混合精度输入，应提升为FP32计算)
  // out:  tensor [2, 3, 5], DT_FLOAT
  // cubeMathType: 0 (ALLOW_FP32_DOWN_PRECISION)
  aclnnBatchMatMulGetWorkspaceSize(self, mat2, out, 0, &workspaceSize, &executor);
  ```
- **预期异常**: 应正常执行，self 自动提升为 FP32 后进行矩阵乘法。实际行为：`CheckDtypeValid` 在第 113 行直接返回 `false`，导致 `CheckBmmOp` 返回 `ACLNN_ERR_PARAM_INVALID`，拒绝合法的混合精度输入。原始代码仅打印 warning 并继续执行。
