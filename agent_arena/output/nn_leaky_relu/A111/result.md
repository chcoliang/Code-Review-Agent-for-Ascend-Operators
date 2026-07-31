**Bug**: 第61-63行 `CheckShape` 函数缺少 `OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false)` 输入输出 shape 一致性检查。当输出 shape 与输入不一致时不会报错，ViewCopy 将导致越界写入或数据错位。

**触发输入**: 传入 self shape 为 [2, 3]，out shape 为 [6]（元素数相同但 shape 不同），或 out shape 为 [4, 3]（元素数不同）。
