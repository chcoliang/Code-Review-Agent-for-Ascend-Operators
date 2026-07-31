# A59 BatchMatMul Code Review

## Bug: CheckMathType 中 promoteType 未使用输入 dtype 判断，恒为 FP16

| 项目 | 内容 |
|------|------|
| **位置** | `CheckMathType` 函数, 第 244-249 行 |
| **描述** | 函数计算了 `selfFloat` 和 `mat2Float` 两个布尔变量用于判断输入是否为 FP32，但随后 `promoteType` 直接硬编码为 `DataType::DT_FLOAT16`，完全忽略了这两个变量。正确实现应为 `auto promoteType = selfFloat \|\| mat2Float ? DataType::DT_FLOAT : DataType::DT_FLOAT16;`。错误导致：当输入为 FP32 时，promoteType 应为 DT_FLOAT，对 `CheckCubeMathTypeForMm` 的调用应使用 FP32 的规则来校验 cubeMathType 的合法性。当前始终以 FP16 的规则校验，可能错误地允许或拒绝某些 cubeMathType 值。例如，FP32 输入下使用 `KEEP_DTYPE` 应该合法（保持 FP32 精度），但以 FP16 规则校验时可能产生不同结论。 |
| **触发输入** | `self`: dtype=DT_FLOAT, shape=[2,4,8]; `mat2`: dtype=DT_FLOAT, shape=[2,8,6]; `out`: dtype=DT_FLOAT, shape=[2,4,6]; `cubeMathType`=KEEP_DTYPE |
| **预期异常** | 应使用 DT_FLOAT 作为 promoteType 进行 cubeMathType 校验（FP32 + KEEP_DTYPE 为合法组合）。实际以 DT_FLOAT16 作为 promoteType 校验，可能导致合法的 cubeMathType 被错误拒绝并返回 `ACLNN_ERR_PARAM_INVALID`，或非法的组合被错误放行。 |
