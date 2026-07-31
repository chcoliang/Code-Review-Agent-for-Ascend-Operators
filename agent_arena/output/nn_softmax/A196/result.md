# A196 代码审查报告 - softmax_v2_base.h

## Bug 1: CopyIn 多行版本缺失 DataCopyPad 调用导致数据未搬运

- **位置**: 第 244-259 行（特别是第 258-259 行函数末尾）
- **类型**: 内存访问/功能缺失
- **严重程度**: 高
- **描述**: `CopyIn` 的多行版本（接受 rowSize, colSize, dstStride, srcStride 参数的重载）中，函数设置了 `DataCopyExtParams params` 和 `DataCopyPadExtParams<T> padParams`，但**未调用 `DataCopyPad(dstTensor, srcTensor, params, padParams)`**。函数在设置 `padParams.isPad = false` 后直接结束，导致数据根本没有从 Global Memory 搬运到 UB（Local Memory）。对比 A195 版本的相同函数（第 259 行），其正确调用了 `DataCopyPad`。
- **触发条件**: 任何使用非最后轴（non-last axis）reduce 的 softmax 场景都会触发此 bug，因为非最后轴场景需要按行搬运数据（rowSize > 1 且有 stride）。此时 UB 中的数据为未初始化的随机值，计算结果完全错误。
- **测试方案**: 
  1. 使用 shape=[4, 8, 16]，axis=1 运行 softmax，验证输出是否全为随机值/零
  2. 对比 A195 相同输入的正确输出
  3. 任何 rowSize > 1 的多行搬运场景（即非连续内存的 reduce 轴）均应验证

---

## 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 244-259 | 内存访问/功能缺失 | 高 | 多行CopyIn函数缺少DataCopyPad调用，数据未搬运到UB |
