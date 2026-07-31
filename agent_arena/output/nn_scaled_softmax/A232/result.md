# A232 代码审查报告

**文件**: `scaled_masked_softmax_v2.h`

---

## Bug 1: CopyXIn 中 DataCopy srcStride 参数错误

- **位置**: 第 127 行
- **类型**: 参数错误
- **严重程度**: 高
- **描述**: `DataCopyExtParams` 中的 `srcStride`（单行拷贝长度）被设置为 `(tilingData.width + 1) * sizeof(T)`，多加了 1。正确值应为 `tilingData.width * sizeof(T)`，即每行实际数据的字节数。多加 1 会导致每行从 Global Memory 多读取一个元素的偏移，使得：
  1. 第一行多读 1 个元素的数据（可能越界）。
  2. 多行拷贝时每行起始地址偏移错误，导致数据错位，后续所有行的数据都向后偏移 `sizeof(T)` 字节。
- **触发条件**: 任何多行输入（height > 1 或单次迭代处理多行）时均会触发数据错位。即使 height=1，也会多读一个元素可能导致越界。
- **测试方案**: 
  1. 使用 batch=1, channel=1, height=2, width=32 的输入，对比输出与参考实现的结果差异。
  2. 设置 width 为该类型对齐块大小的整数倍，验证多行数据是否正确对齐。
  3. 检查是否有内存越界访问（使用 width 接近页面边界的 tensor）。

---

## 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 127 | 参数错误 | 高 | DataCopyExtParams 的 srcStride 多加了 1，导致多行拷贝数据错位 |
