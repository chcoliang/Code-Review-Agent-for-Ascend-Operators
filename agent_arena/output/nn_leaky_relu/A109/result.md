**Bug**: 第100行 `auto selfContiguous = self;` 缺少 `l0op::Contiguous` 调用，直接使用原始输入 tensor 而未转换为连续存储。当输入为非连续 tensor 时（如 transpose/slice 后），kernel 将按连续布局读取不连续的内存，产生错误计算结果。

**触发输入**: 传入经过 transpose 操作后的非连续 tensor，例如 shape [4,3] 经 transpose 得到 stride 不连续的 [3,4] tensor。
