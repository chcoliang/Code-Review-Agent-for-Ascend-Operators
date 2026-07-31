complex32分支（第47-50行）使用`MulDag::MulComplex32Op<complex32, complex64>`模板参数为复数类型，而模板定义中期望的是存储类型`<int32_t, int64_t>`，导致CopyInBrc和Cast操作的类型不匹配，编译错误或运行时数据解释错误。触发条件：complex32类型tensor做Mul。
