# A56 BatchMatMul Code Review

## Bug: CheckShape 缺少对 selfTensor 维度的校验

| 项目 | 内容 |
|------|------|
| **位置** | `CheckShape` 函数, 约第 196-199 行 |
| **描述** | 函数开头使用 `OP_CHECK_WRONG_DIMENSION` 对 otherTensor 和 outTensor 进行维度下限检查（要求 >= SHAPE_LIMIT=3），但缺少对 `selfTensor` 的同类检查。正确实现应包含 `OP_CHECK_WRONG_DIMENSION(selfTensor, SHAPE_LIMIT, return false);`。缺失此检查后，1D 或 2D 的 self tensor 可以通过 shape 校验，后续通过 `self[FIRST_DIM]` 等索引访问时，对于 2D tensor `FIRST_DIM=0` 仍可访问，但 `selfDimNum - PENULTIMATE_DIM` 对 2D 为 0，可能导致语义错误；更严重的是后续 `ProcessEmptyTensor` 中直接用 `[0],[1],[2]` 索引构造输出 shape 时会越界。 |
| **触发输入** | `self`: dtype=DT_FLOAT16, shape=[4,8]（2D tensor）; `mat2`: dtype=DT_FLOAT16, shape=[2,8,6]; `out`: dtype=DT_FLOAT16, shape=[2,4,6]; `cubeMathType`=0 |
| **预期异常** | 应在 CheckShape 中返回 `ACLNN_ERR_PARAM_INVALID`，报错提示 selfTensor 维度不满足要求。实际会绕过维度检查，后续计算中出现 shape 不匹配或越界访问。 |
