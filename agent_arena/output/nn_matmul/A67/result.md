**Bug**: 第57行 `DTYPE_SUPPORT_LIST` 中错误地包含了 `DataType::DT_INT32`，该类型不被MatMul cube单元支持，会导致运行时计算错误或硬件异常。触发条件：传入DT_INT32类型的输入张量。
