# Softmax V2 Base 代码审查报告

## Bug 列表

### Bug 1: CopyIn 单行重载 blockLen Off-by-One 错误

- **位置**: 第 269 行，`SoftmaxV2OpsBase::CopyIn` 单参数重载（single-row版本）
- **类型**: Off-by-One 错误
- **严重程度**: 严重 (Critical)
- **描述**:
  ```cpp
  params.blockLen = (rowSize - 1) * sizeof(T);
  ```
  `blockLen` 被设置为 `(rowSize - 1) * sizeof(T)`，导致实际拷贝的数据比预期少一个元素。正确应为 `rowSize * sizeof(T)`。对比同文件中语义对等的 `CopyOut` 单行重载（第 283 行）使用的是 `rowSize * sizeof(T)`，可确认此处为笔误。

  DataCopyPad 的 `blockLen` 参数表示单个 block 的拷贝字节数。此处 `blockCount=1`，`blockLen` 即为总拷贝量。减 1 导致最后一个元素未从 GM 拷入 UB。

- **触发条件**:
  - 当调用方使用单行 `CopyIn` 重载从 Global Memory 拷贝数据到 Local Memory 时必然触发。
  - 任何 `rowSize >= 1` 的情况下，最后一个元素都不会被拷贝。
  - 若 `rowSize == 1`，则 `blockLen = 0`，完全不拷贝任何数据。

- **测试方案**:
  1. 构造输入 tensor，元素值设为递增序列 `[0, 1, 2, ..., N-1]`。
  2. 调用单行 `CopyIn(localTensor, globalTensor, N)`。
  3. 验证 localTensor 中第 `N-1` 个元素是否等于 `N-1`（预期失败，该位置为未初始化值或旧数据）。
  4. 特别测试 `rowSize=1` 的边界情况，确认是否有任何数据被拷贝。

- **修复建议**:
  ```cpp
  // 修复前
  params.blockLen = (rowSize - 1) * sizeof(T);
  // 修复后
  params.blockLen = rowSize * sizeof(T);
  ```

---

### Bug 2: CopyIn 多行重载 srcStride 单位不一致风险

- **位置**: 第 253 行，`SoftmaxV2OpsBase::CopyIn` 多参数重载
- **类型**: DataCopy 参数计算错误（潜在）
- **严重程度**: 中等 (Medium)
- **描述**:
  ```cpp
  params.srcStride = srcStride * sizeof(T) - params.blockLen;
  ```
  `DataCopyExtParams::srcStride`（GM 侧）表示连续 block 之间的**间隔字节数**（gap）。此处计算逻辑假设 `srcStride` 参数为行间总跨度（以元素为单位）。若调用方传入的 `srcStride` 语义不是"行起始到下一行起始的总元素数"，而是"block 之间的间隔元素数"，则会产生双重减法错误。

  同时，对应的 `dstStride` 计算：
  ```cpp
  params.dstStride = (dstStride * sizeof(T) - ops::Aligned(blockLen, GetUbBlockSize())) / GetUbBlockSize();
  ```
  当 `dstStride * sizeof(T) < ops::Aligned(blockLen, GetUbBlockSize())` 时（即 dstStride 不足以容纳对齐后的 blockLen），会产生负值被截断为大正数（uint 类型），导致内存越界。

- **触发条件**:
  - 当 `colSize` 未对齐到 32 字节且 `dstStride == colSize` 时，`dstStride * sizeof(T)` 可能小于 `Aligned(blockLen, 32)`，导致无符号整数下溢。
  - 依赖调用方保证 `dstStride` 已包含对齐 padding。

- **测试方案**:
  1. 构造 `colSize` 不对齐到 32/sizeof(T) 的场景（如 T=half, colSize=7）。
  2. 传入 `dstStride = colSize`（不含对齐padding）。
  3. 检测是否发生 UB 内存越界写入。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第 269 行 `CopyIn` 单行重载 | Off-by-One | 严重 | `blockLen` 少算一个元素，最后一个元素未被拷贝 |
| 2 | 第 253-256 行 `CopyIn` 多行重载 | DataCopy 参数计算 | 中等 | `dstStride` 在非对齐场景下可能无符号下溢 |
