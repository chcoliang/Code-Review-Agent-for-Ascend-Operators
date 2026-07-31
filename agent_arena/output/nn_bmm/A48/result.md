# A48 BatchMatMul Code Review

## Bug: DTYPE_SUPPORT_LIST 缺少 DT_BF16 类型

- **位置**: 第 81-82 行, `DTYPE_SUPPORT_LIST` 定义
- **描述**: `DTYPE_SUPPORT_LIST` 仅包含 `{DT_FLOAT, DT_FLOAT16}`，缺少 `DT_BF16`。在 Ascend 910B 等支持 BF16 的平台上，`CheckDtypeValid` 函数中 `bf16flag` 为 `true`，选用 `DTYPE_SUPPORT_LIST` 作为合法类型列表。但由于列表中没有 `DT_BF16`，BF16 输入会被 `OP_CHECK_DTYPE_NOT_SUPPORT` 错误拒绝，导致本应支持的 BF16 BatchMatMul 计算无法执行。
- **触发输入**:
  ```cpp
  // 在 Ascend 910B 上:
  // self: tensor [2, 3, 4], DT_BF16
  // mat2: tensor [2, 4, 5], DT_BF16
  // out:  tensor [2, 3, 5], DT_BF16
  // cubeMathType: 0 (ALLOW_FP32_DOWN_PRECISION)
  aclnnBatchMatMulGetWorkspaceSize(self, mat2, out, 0, &workspaceSize, &executor);
  ```
- **预期异常**: 应正常执行 BF16 BatchMatMul。实际行为：返回 `ACLNN_ERR_PARAM_INVALID`，报错 dtype 不支持，拒绝合法的 BF16 输入。
