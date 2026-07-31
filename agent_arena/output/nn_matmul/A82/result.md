**Bug**: 第49行 `GlobalTensor<A_T> cGlobal_` 使用了输入A的类型 `A_T` 而非输出C的类型 `C_T`，当输入输出dtype不同时（如FP16输入FP32输出），写出结果时类型宽度错误导致数据截断或内存越界。触发条件：A_TYPE=FP16, C_TYPE=FP32的matmul运算。
