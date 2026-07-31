# A54 BatchMatMul Code Review

## Bug: aclnnBatchMatMulGetWorkspaceSize 缺少空 tensor 提前返回逻辑

| 项目 | 内容 |
|------|------|
| **位置** | `aclnnBatchMatMulGetWorkspaceSize` 函数, 约第 1008-1009 行 (CheckParamsV2 之后，CreateBatchMatmulGraphImpl 之前) |
| **描述** | 在参数校验通过后，缺少对空 tensor 场景的提前返回判断。正确实现应在 CheckParamsV2 之后检查 `CheckBmmResIsEmpty(self, mat2)`，若为空则直接设置 `*workspaceSize = 0` 并返回 `ACLNN_SUCCESS`。当前实现直接进入 `CreateBatchMatmulGraphImpl`，而该函数内部存在逻辑缺陷（空 tensor 分支被正常分支覆盖），导致空 tensor 会走入正常计算图执行路径 `ExecBmmOpV2`，对 B=0 或 M=0 或 N=0 的 tensor 执行无意义的计算，可能导致访问非法维度或内部算子报错。 |
| **触发输入** | `self`: dtype=DT_FLOAT16, shape=[0,4,8]; `mat2`: dtype=DT_FLOAT16, shape=[0,8,6]; `out`: dtype=DT_FLOAT16, shape=[0,4,6]; `cubeMathType`=0 |
| **预期异常** | 应正常返回 `ACLNN_SUCCESS` 且 `*workspaceSize = 0`（空 tensor 无需计算）。实际可能触发内部空指针或 shape 非法的运行时错误。 |
