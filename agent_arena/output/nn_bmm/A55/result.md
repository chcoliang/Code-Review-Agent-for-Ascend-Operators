# A55 BatchMatMul Code Review

## Bug: ExecBmmOpWithBias 中非转置 self 缺少 Contiguous 处理

| 项目 | 内容 |
|------|------|
| **位置** | `ExecBmmOpWithBias` 函数, 约第 864-867 行 |
| **描述** | 在处理 `contiguousSelf` 时，当 `transposeSelf` 为 true 时通过 `CreateView` 处理，但当 `transposeSelf` 为 false 时缺少 `else` 分支调用 `l0op::Contiguous(self, executor)`。这意味着当 self tensor 既非内轴转置、又非连续存储（如 stride 不规则的 view/slice）时，`contiguousSelf` 保持为 `reformatSelf`（仅做了 ReFormat），其底层存储仍然是非连续的。将非连续 tensor 直接送入 Cube 计算单元会导致数据读取错乱，产出错误的矩阵乘结果。 |
| **触发输入** | `self`: dtype=DT_FLOAT16, shape=[2,4,8], strides=[64,16,2]（非连续，如通过 slice 得到）; `mat2`: dtype=DT_FLOAT16, shape=[2,8,6], 连续; `out`: dtype=DT_FLOAT16, shape=[2,4,6]; `cubeMathType`=0 |
| **预期异常** | 应将非连续 self 通过 `l0op::Contiguous` 转为连续后再计算。实际会使用非连续数据直接参与 Cube 计算，导致计算结果静默错误（无异常抛出但结果不正确）。 |
