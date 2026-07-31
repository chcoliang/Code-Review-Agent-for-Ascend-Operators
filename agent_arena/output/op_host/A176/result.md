FP16分支（第119-123行）错误使用`MulOp<float>::OpDag`而非`MulXfp16Op<half>::OpDag`，缺少FP16升精度到FP32再降回FP16的Cast操作，导致tiling与kernel DAG不匹配，FP16乘法计算精度错误或运行时异常。触发条件：两个FP16 tensor做Mul运算。
