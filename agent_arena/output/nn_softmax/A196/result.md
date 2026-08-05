# Softmax V2 Base 代码审查报告

## Bug 列表

### Bug 1: CopyIn 多行变体缺少 DataCopyPad 调用

- **位置**: 第 244-258 行，`CopyIn(dstTensor, srcTensor, rowSize, colSize, dstStride, srcStride)` 函数
- **类型**: DataCopy 缺失
- **严重程度**: 致命 (Critical)
- **描述**: 该函数准备了 `DataCopyExtParams params` 和 `DataCopyPadExtParams<T> padParams` 的所有参数，但函数体末尾没有调用 `DataCopyPad(dstTensor, srcTensor, params, padParams)`。对比同文件中单行 CopyIn（第 262-272 行）正确调用了 `DataCopyPad`，以及多行 CopyOut（第 286-299 行）也正确调用了 `DataCopyPad`，可以确认这是一处遗漏。该函数实际上不会从 Global Memory 拷贝任何数据到 Local Memory（UB），导致后续计算使用的是 UB 中的未初始化/残留数据。
- **触发条件**: 任何使用多行（rowSize > 1 且需要 stride）方式从 GM 搬入数据到 UB 的场景都会触发。典型场景为 softmax 沿非最后一个轴进行归约时，需要按 stride 跨行读取输入 tensor。
- **修复方案**: 在第 258 行 `padParams.isPad = false;` 之后添加:
  ```cpp
  DataCopyPad(dstTensor, srcTensor, params, padParams);
  ```
- **测试方案**: 
  1. 构造多维输入 tensor（如 shape=[4, 8, 16]），对非最后轴（如 axis=1）执行 softmax
  2. 验证输出结果与 numpy/pytorch 参考实现一致
  3. 对比修复前后的输出，修复前输出应为随机值或全零

---

### Bug 2: CopyIn 多行变体中 dstStride 与 srcStride 单位不对称（潜在问题）

- **位置**: 第 253-256 行
- **类型**: 内存访问 / 参数计算
- **严重程度**: 中等 (Medium)
- **描述**: `params.srcStride` 以字节为单位计算 GM 侧的 block 间隙（`srcStride * sizeof(T) - blockLen`），而 `params.dstStride` 以 UB Block（32 字节）为单位计算 UB 侧的 block 间隙（`(dstStride * sizeof(T) - Aligned(blockLen, UbBlockSize)) / GetUbBlockSize()`）。这种不对称性依赖于 `DataCopyPad` API 对 GM→UB 方向的特定约定：GM 侧 stride 用字节，UB 侧 stride 用 32 字节 block 数。如果 API 版本或平台实现对 stride 单位有不同约定，该计算将导致数据排布错误。同时，当 `dstStride * sizeof(T)` 不能被 `GetUbBlockSize()` 整除时，整除运算会丢失精度。
- **触发条件**: 当 dstStride（UB 侧行间距）不是 32 字节对齐的倍数时，整除截断可能导致行数据重叠或间隙计算错误。
- **测试方案**:
  1. 设置 colSize 使得 `blockLen` 不是 32 字节的倍数（如 colSize=5, T=half, blockLen=10）
  2. 设置 dstStride 使得 dstStride*sizeof(T) 不能被 32 整除
  3. 检查 UB 中多行数据排布是否正确

---

### Bug 3: CopyUB2UB 对齐拷贝可能越界写入

- **位置**: 第 301-308 行，`CopyUB2UB` 函数
- **类型**: 内存访问越界
- **严重程度**: 低 (Low)
- **描述**: 函数使用 `ops::Aligned(count, GetUbBlockSize() / sizeof(float))` 将拷贝长度向上对齐到 8 个 float（32 字节）边界。如果 `dstTensor` 的分配大小恰好等于 `count` 个 float 且 count 不是 8 的倍数，则对齐后的拷贝长度会超出 dst 的有效范围，写入相邻内存区域。在 UB-to-UB DataCopy 中，硬件要求 32 字节对齐传输，因此 dst buffer 后方可能存在其他有效数据被覆盖。
- **触发条件**: count 不是 8 的倍数，且 dstTensor 后方紧邻其他正在使用的 UB buffer。
- **测试方案**:
  1. 分配两个紧邻的 UB buffer A 和 B
  2. 对 A 执行 CopyUB2UB，count 不对齐（如 count=5）
  3. 检查 B 的前几个元素是否被意外修改

---

### Bug 4: LastReduceSumSmallR 中 ReduceSum 使用 pFull 掩码时依赖寄存器残留值

- **位置**: 第 481-487 行（`rSize > VL_FP32` 分支）
- **类型**: 数据正确性 / 寄存器状态依赖
- **严重程度**: 低 (Low)
- **描述**: 在 `rSize > VL_FP32 && rSize <= 2*VL_FP32` 的分支中，代码先用部分掩码 `pMask`（覆盖 `rSize - VL_FP32` 个元素）执行 Add+Copy MERGING，然后对 aReg 使用全掩码 `pFull` 调用 ReduceSum。这依赖于 aReg 在 MERGING 之后未被修改的位置仍保持第一次 DataCopy 加载的值（即 src0 的前 VL_FP32 个元素）。虽然算法设计上是正确的（利用 MERGING 语义拼接两部分数据），但如果编译器对寄存器生命周期做优化（如在 Add ZEROING 期间复用 aReg 的物理寄存器），可能导致 aReg 的非掩码位置包含垃圾值。
- **触发条件**: 特定编译器优化级别下，aReg 的非活跃 lane 被复用或清除。
- **测试方案**:
  1. 设置 rSize = VL_FP32 + 1（仅超出一个元素）
  2. 填充已知数据，验证 ReduceSum 结果是否等于所有 rSize 个元素的精确和
  3. 在不同优化级别下对比结果

---

### Bug 5: NlastBroadcastMul 末尾向量加载可能越界读取

- **位置**: 第 427-431 行
- **类型**: 内存访问越界（读取）
- **严重程度**: 低 (Low)
- **描述**: 当 `aSize` 不是 `VL_FP32` 的整数倍时，最后一次迭代的 `DataCopy(aReg, src0 + i*VL_FP32 + j*aSize)` 和 `DataCopy(bReg, src1 + i*VL_FP32)` 会从内存中加载完整的一个向量寄存器长度（VL_FP32 个 float），读取范围超出 tensor 的有效数据区域。虽然掩码确保只有有效位置被写出，且 UB 通常有对齐 padding，但如果 src1 tensor 分配恰好位于 UB 末尾边界附近，可能触发硬件异常。
- **触发条件**: aSize 不是 VL_FP32 的倍数，且源 tensor 分配在 UB 空间末尾附近无 padding。
- **测试方案**:
  1. 设置 aSize = VL_FP32 + 1，将 src1 tensor 分配在 UB 可用空间的最末端
  2. 检查是否触发 bus error 或读取越界异常

---

## 汇总表

| 编号 | 位置 (行号) | Bug 类型 | 严重程度 | 简要描述 |
|------|-------------|----------|----------|----------|
| 1 | 244-258 | DataCopy 缺失 | 致命 | 多行 CopyIn 函数未调用 DataCopyPad，数据未实际拷贝 |
| 2 | 253-256 | 参数计算 | 中等 | srcStride/dstStride 单位不对称，依赖特定 API 约定 |
| 3 | 301-308 | 内存越界写入 | 低 | 对齐拷贝长度可能超出 dst buffer 有效范围 |
| 4 | 481-487 | 数据正确性 | 低 | ReduceSum 依赖寄存器非活跃 lane 的残留值 |
| 5 | 427-431 | 内存越界读取 | 低 | 末尾向量加载超出 tensor 有效数据区域 |

## 总结

最关键的问题是 **Bug 1**：多行 `CopyIn` 函数完全无效，构造了所有 DataCopy 参数却遗漏了实际的 `DataCopyPad` 调用。这会导致所有依赖多行数据搬入的计算路径产生错误结果（使用 UB 中的垃圾数据）。修复方法是在函数末尾添加 `DataCopyPad(dstTensor, srcTensor, params, padParams);`。其余问题为潜在的边界条件风险，在特定参数组合下可能触发。
