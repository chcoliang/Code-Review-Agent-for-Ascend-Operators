**Bug:** 第150行 `SwishNegOne` 结构体中 `using OpCopyOut = Bind<Vec::CopyOut<float>, Placeholder::Out0<float>, OpResult>` 输出类型硬编码为 float，对于 half/bfloat16 类型 T，CopyOut 和 Out0 应使用 T 而非 float。这导致 fp16/bf16 场景下输出按 float（4字节）写入本应为 2 字节的输出 buffer，造成内存越界写和数据损坏。

**触发输入:** 使用 dtype=FLOAT16 或 BF16 输入且 scale=-1 的场景触发 SwishNegOne 路径。
