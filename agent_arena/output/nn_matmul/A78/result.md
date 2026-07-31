**Bug**: 第24行 x1 Input的第一个DataType为 `ge::DT_INT8`（应为 `ge::DT_FLOAT16`），导致INT8×FP16的非法dtype组合被注册为合法，Ascend910B上选中该规则时产生计算错误或下发失败。触发条件：x1=INT8, x2=FP16输入在ascend910b上调用MatMulV3。
