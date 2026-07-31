`InnerTypeToComplexType`中`DT_FLOAT16`错误地映射为`DT_COMPLEX64`（第69行），正确应为`DT_COMPLEX32`，导致FP16与复数类型混合运算时类型提升错误，计算结果dtype不符合预期。触发条件：在ASCEND910_95上用FP16 tensor与复数scalar做Mul运算。
