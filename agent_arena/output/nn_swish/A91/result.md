**Bug:** 第102行 `auto selfContiguous = self;` 缺少 `l0op::Contiguous()` 调用，输入张量未被转为连续内存布局。当输入为非连续张量（如 transpose/slice 后的视图）时，后续计算将按连续内存方式读取不连续数据，导致计算结果完全错误。

**触发输入:** 传入经过 transpose 或 slice 操作的非连续（non-contiguous）张量作为 self 参数。
