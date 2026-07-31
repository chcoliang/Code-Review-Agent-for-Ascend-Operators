# A49 BatchMatMul Code Review

## Bug: DTYPE_SUPPORT_LIST 错误包含 DT_INT32 类型

- **位置**: 第 81-82 行, `DTYPE_SUPPORT_LIST` 定义
- **描述**: `DTYPE_SUPPORT_LIST` 包含了 `op::DataType::DT_INT32`，即 `{DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT32}`。BatchMatMul 是 Cube 算子，Ascend 910B 的 Cube 单元不支持 INT32 矩阵乘法运算。INT32 输入通过 dtype 校验后，会进入后续的 Cast/TransData/BatchMatMul 计算流程，在 Cube 执行时产生未定义行为或硬件错误。
- **触发输入**:
  ```cpp
  // 在 Ascend 910B 上:
  // self: tensor [2, 3, 4], DT_INT32
  // mat2: tensor [2, 4, 5], DT_INT32
  // out:  tensor [2, 3, 5], DT_INT32
  // cubeMathType: 0
  aclnnBatchMatMulGetWorkspaceSize(self, mat2, out, 0, &workspaceSize, &executor);
  ```
- **预期异常**: 应在 dtype 校验阶段返回 `ACLNN_ERR_PARAM_INVALID`，拒绝 INT32 输入。实际行为：dtype 校验通过，INT32 数据进入 Cube 计算路径，导致下游 `l0op::BatchMatMulNd` 等接口执行失败或产生错误结果（Cube 不支持 INT32 matmul）。
