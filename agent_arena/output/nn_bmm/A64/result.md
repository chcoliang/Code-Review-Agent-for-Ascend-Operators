# A64 审查结果: kernel层GlobalTensor类型错误

## Bug

| 项目 | 内容 |
|------|------|
| **文件** | `batch_mat_mul_v3.h` |
| **位置** | 第57行, `BatchMatMulMultiBatchKernel` 类的成员变量声明 |
| **问题代码** | `GlobalTensor<A_T> cGlobal_;` |
| **描述** | 输出矩阵C的GlobalTensor类型声明错误，使用了 `A_T`（输入矩阵A的元素类型）而非 `C_T`（输出矩阵C的元素类型）。正确声明应为 `GlobalTensor<C_T> cGlobal_;`。对比同文件中 `BatchMatMulCommonKernel` 类（第299行）正确使用了 `GlobalTensor<C_T> cGlobal_;`。当A和C的dtype不同时（如 x1=FP16, y=FP32 或 x1=BF16, y=FP32），`A_T != C_T`，类型链将发生断裂。 |
| **触发条件** | 当BatchMatMulV3的输入dtype与输出dtype不同时触发，例如：x1=FP16, x2=FP16, y=FP32（混合精度matmul），或 x1=BF16, x2=BF16, y=FP32。此时 `A_T=half/bfloat16_t` 而 `C_T=float`，cGlobal_的元素类型与实际输出数据类型不匹配。 |
| **预期异常** | 1) 编译阶段：当模板实例化时 `A_T != C_T`，`SetGlobalBuffer` 接收 `reinterpret_cast<__gm__ C_T*>(cGM)` 但赋值给 `GlobalTensor<A_T>`，导致编译报错（类型不匹配）或需要隐式转换。2) 若编译通过（如通过强制转换）：运行时写出数据按 `A_T` 的字节宽度计算偏移，FP16(2B) vs FP32(4B) 会导致输出地址计算错误，写入位置偏移一半，产生数据覆盖和输出结果完全错乱。 |
