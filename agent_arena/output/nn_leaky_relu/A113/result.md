**Bug**: 第107-111行缺少 `l0op::Cast(output, out->GetDataType(), ...)` 调用，直接将 LeakyRelu 输出传入 ViewCopy 而未做数据类型转换。当输出 tensor 的 dtype 与计算结果 dtype 不同时，数据将以错误类型解释写入。

**触发输入**: 输入 self 为 FLOAT 类型，输出 out 为 FLOAT16 类型，写入的数据将是未转换的 FLOAT 原始字节。
