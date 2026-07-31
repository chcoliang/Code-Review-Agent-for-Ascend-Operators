**Bug**: 第95行处理 DT_FLOAT16 时未设置 `dType` 变量（保持默认值0），且第103行处理 DT_BF16 时错误使用了 `LeakyReluCastDag<half>` 模板（应为 `LeakyReluCastDag<bfloat16_t>`）。BF16 数据将被错误地按 FP16 格式解释进行 CopyIn 和 Cast，导致计算结果完全错误。

**触发输入**: 传入 dtype 为 BF16 的 tensor，kernel 将以 half 类型读取 bfloat16 数据，产生乱码结果。
