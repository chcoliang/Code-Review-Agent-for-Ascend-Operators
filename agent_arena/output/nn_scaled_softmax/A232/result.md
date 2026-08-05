# ScaledMaskedSoftmaxV2 代码审查报告

## Bug 列表

### Bug 1: CopyXIn 中 DataCopyPad 的 blockLen 参数多加1

- **位置**: 第 127 行，`CopyXIn` 函数
- **类型**: DataCopy 参数错误（off-by-one）
- **严重程度**: 高
- **描述**: `DataCopyExtParams` 中 `blockLen` 设为 `(tilingData.width + 1) * sizeof(T)`，但实际 GM 中每行有效数据宽度为 `tilingData.width` 个元素。正确值应为 `tilingData.width * sizeof(T)`。多出的 `+1` 导致每行从 GM 多读取一个元素，读入下一行首元素或越界数据。同时，`paddingNum` 是基于 `width` 对齐计算的，`blockLen` 多读一个元素后与 padding 配合失配，导致 UB 中数据排列错乱。
- **触发条件**: 任何情况下都会触发。每一行都会多读1个元素，在多行场景下第2行起数据会发生错位；在最后一行可能导致 GM 越界访问。
- **修复建议**: 将 `(tilingData.width + 1) * sizeof(T)` 改为 `tilingData.width * sizeof(T)`。
- **测试方案**: 
  1. 构造 width 不对齐32B的小张量（如 batch=1, channel=1, height=2, width=13），对比输出与参考实现。
  2. 检查第二行及之后行的数据是否正确，验证是否有错位。
  3. 使用边界 width 使最后一行刚好到 GM 边界，检测是否有非法内存访问。

---

### Bug 2: CopyOut 中 DataCopyPad 缺少 srcStride 导致输出数据错位

- **位置**: 第 280-282 行，`CopyOut` 函数
- **类型**: DataCopy 参数错误（stride 缺失）
- **严重程度**: 高
- **描述**: CopyOut 使用 `DataCopyPad` 将 UB 中的 `yTensor` 写回 GM。UB 中每行占 `padLineNum` 个元素（含 padding 对齐），但 `DataCopyExtParams` 中 `srcStride=0`（即连续读取）。当 `padLineNum > xWidth` 时（几乎所有需要 padding 的场景），从第2行起会从错误偏移读取数据——读到的是上一行的 padding 区域而非下一行的有效数据。`blockLen = xWidth * sizeof(T)` 是正确的（只写有效宽度），但 srcStride 应设为 `(padLineNum - xWidth) * sizeof(T) / 32` 以跳过每行尾部的 padding。
- **触发条件**: 当 `padLineNum != xWidth`（即 width 不满足对齐要求需要 padding）且 `linePerIter > 1` 时触发。仅单行迭代时不会出错。
- **修复建议**: 计算 srcStride 为 `(tilingData.padLineNum * sizeof(T) - this->xWidth * sizeof(T)) / 32`，填入 DataCopyExtParams 的 srcStride 字段。
- **测试方案**:
  1. 构造 width 需要 padding 对齐且 height > 1 的输入（如 width=100, half类型，padLineNum=128）。
  2. 对比输出 GM 中第2行及后续行数据与预期值。
  3. 验证 padding 宽度为0时（width 恰好对齐）结果正确作为对照。

---

### Bug 3: MaskOffset::GetOffset 计算逻辑语义颠倒

- **位置**: 第 48 行，`GetOffset` 函数
- **类型**: 内存访问偏移计算错误
- **严重程度**: 中
- **描述**: 函数参数命名为 `realBatch`、`realChannel`、`realLine`，但实际传入的是步长（stride）值：`maskBatchOffset`（batch间距）、`maskChannelOffset`（channel间距）、`tilingData.maskWidth`（行间距）。计算 `batchOffset * realBatch + channelOffset * realChannel + lineOffset * realLine` 在 `nStep!=1` 或 `cStep!=1` 的 broadcast 场景下，若 `batchOffset` 值超出 mask 实际 batch 数（因为 MaskOffset 中 batchOffset 按输入 x 的 batch 递增，而非 mask 的 batch），会导致 mask 的 GM 偏移越界。但在 `nStep=1, cStep=1`（无 broadcast）场景下结果恰好正确。
- **触发条件**: 当 mask 的 batch/channel 维度与输入 x 存在 broadcast 关系（nStep>1 或 cStep>1），且跨 batch 迭代时，偏移计算超出 mask 实际范围。
- **测试方案**:
  1. 构造 mask broadcast 场景：x shape=[4,8,32,64], mask shape=[1,1,32,64]（nStep=4, cStep=8）。
  2. 验证多 batch/channel 下 mask 是否正确复用，而非越界访问。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第127行 CopyXIn | DataCopy参数错误 | 高 | blockLen 多加1，每行多读一个元素导致数据错位和潜在越界 |
| 2 | 第280-282行 CopyOut | DataCopy参数错误 | 高 | srcStride=0未跳过UB中padding，多行输出数据错位 |
| 3 | 第48行 GetOffset | 内存访问偏移计算 | 中 | broadcast场景下batchOffset未取模，mask GM偏移可能越界 |
