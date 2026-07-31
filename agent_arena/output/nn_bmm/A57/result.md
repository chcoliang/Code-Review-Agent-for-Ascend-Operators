# A57 BatchMatMul Code Review

## Bug: CheckShape 缺少 K 维度匹配校验（self 的最后一维 != mat2 的倒数第二维）

| 项目 | 内容 |
|------|------|
| **位置** | `CheckShape` 函数, 约第 214-215 行（应位于 dimNum < 2 检查之后） |
| **描述** | 缺少对矩阵乘法核心约束的校验：`self[selfDimNum - 1] != other[otherDimNum - PENULTIMATE_DIM]`（即 self 的 K 维必须等于 mat2 的 K 维）。正确实现中，若 self 的最后一维与 mat2 的倒数第二维不相等，应报错返回。缺失此检查后，shape 不兼容的矩阵对（如 [2,4,8] @ [2,5,6]，K=8 vs K=5）可通过校验进入 Cube 计算，导致硬件级别的非法内存访问或计算异常。 |
| **触发输入** | `self`: dtype=DT_FLOAT16, shape=[2,4,8]; `mat2`: dtype=DT_FLOAT16, shape=[2,5,6]（K 维不匹配：8 != 5）; `out`: dtype=DT_FLOAT16, shape=[2,4,6]; `cubeMathType`=0 |
| **预期异常** | 应返回 `ACLNN_ERR_PARAM_INVALID`，报错 "self's last dim and mat2's penultimate dim should be same"。实际会绕过 K 维检查，将不兼容 shape 送入底层 Cube 算子，可能导致段错误或 AICORE 异常。 |
