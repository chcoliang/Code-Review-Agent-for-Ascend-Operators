**Bug**: 第24行输入 DataType 列表为 `{DT_INT8, DT_FLOAT16, DT_FLOAT}`，第28行输出为 `{DT_BF16, DT_FLOAT16, DT_FLOAT}`。输入的第一项 DT_INT8 与输出的 DT_BF16 按位置映射，不合理——LeakyReLU 不应接受 INT8 输入且映射到 BF16 输出，应为 DT_BF16 输入对应 DT_BF16 输出。

**触发输入**: 传入 INT8 类型的 tensor，框架将按 INT8→BF16 的错误映射进行调度，kernel 执行时数据解释错误。
