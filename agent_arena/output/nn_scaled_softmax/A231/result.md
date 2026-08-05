# ScaledMaskedSoftmaxV2 代码审查报告

## Bug 列表

### Bug 1: MASK_VAL 常量值错误

- **位置**: 第 22 行 `constexpr float MASK_VAL = 0.0f;`
- **类型**: Kernel 逻辑错误 / Mask 值错误
- **严重程度**: 严重 (Critical)
- **描述**: Scaled Masked Softmax 的核心语义是：对被 mask 的位置填充一个极大负值（如 `-inf` 或 `-10000.0f`），使得经过 softmax（即 `exp(x)/sum(exp(x))`）后这些位置的概率趋近于 0。当前代码将 `MASK_VAL` 设为 `0.0f`，导致被 mask 的位置在 softmax 中对应 `exp(0)=1`，即这些位置获得了非零的注意力权重，完全违背了 masking 的目的。
- **触发条件**: 任何存在 mask=false 位置的输入场景（即实际需要遮蔽的 token 位置），所有调用路径均会触发。例如：attention 中 padding token 的遮蔽、causal mask 等。
- **修复建议**: 将 `MASK_VAL` 修改为负无穷或足够大的负数：
  ```cpp
  constexpr float MASK_VAL = -10000.0f; // 或使用 -std::numeric_limits<float>::infinity()
  ```
- **测试方案**:
  1. 构造输入 x 全为 0，mask 中部分位置为 false（被遮蔽）。
  2. 验证输出中被遮蔽位置的 softmax 值是否趋近于 0。
  3. 验证未遮蔽位置的 softmax 值之和是否趋近于 1。
  4. 与 PyTorch `torch.nn.functional.scaled_dot_product_attention` 中的 masked softmax 对比精度。

---

### Bug 2: sharedBuffer 大小判定条件单位不一致

- **位置**: 第 68-71 行
  ```cpp
  uint64_t tmpBufSize = 32 * 1024; // 32K
  if (bufSize > tmpBufSize) {
      tmpBufSize = 64 * 1024;
  }
  ```
- **类型**: 内存访问 / 潜在越界
- **严重程度**: 中等 (Medium)
- **描述**: `bufSize` 的单位是**元素个数**（`padLineNum * lineHeadIter`，即 float 元素数量），而 `tmpBufSize` 的初始值 `32*1024` 是**字节数**（注释写了 "32K"）。判断条件 `bufSize > tmpBufSize` 将元素数与字节数直接比较，逻辑上不一致。当 `bufSize` 在 8193~32768 元素之间时（对应 32772~131072 字节），SoftMax 高阶 API 可能需要更大的 sharedBuffer，但条件判断未能正确触发扩容，可能导致 SoftMax 内部越界写入。
- **触发条件**: 当 `padLineNum * lineHeadIter` 介于一定范围内、且 SoftMax API 实际所需临时空间超过 32KB 时触发。具体取决于 SoftMax tiling 配置。
- **修复建议**: 统一比较单位，例如：
  ```cpp
  uint64_t bufSizeBytes = bufSize * sizeof(float);
  uint64_t tmpBufSize = 32 * 1024;
  if (bufSizeBytes > tmpBufSize) {
      tmpBufSize = 64 * 1024;
  }
  ```
  或根据 SoftMax API 文档计算实际所需临时空间大小。
- **测试方案**:
  1. 构造 `padLineNum * lineHeadIter` 在 8193~32768 范围内的 tiling 配置。
  2. 运行 kernel 并开启内存越界检测工具（如 CANN 的 overflow detection）。
  3. 验证 SoftMax 计算结果正确性。

---

### Bug 3: GetOffset 函数参数语义与命名不匹配（可维护性缺陷）

- **位置**: 第 46-49 行
  ```cpp
  __aicore__ inline uint64_t GetOffset(uint64_t realBatch, uint64_t realChannel, uint64_t realLine)
  {
      return batchOffset * realBatch + channelOffset * realChannel + lineOffset * realLine;
  }
  ```
- **类型**: 代码可维护性 / 潜在逻辑错误风险
- **严重程度**: 低 (Low)
- **描述**: 参数命名为 `realBatch`、`realChannel`、`realLine`，暗示是维度大小，但实际传入的是 stride（步长）：`maskBatchOffset`、`maskChannelOffset`、`tilingData.maskWidth`。函数内部计算 `index * stride` 是正确的，但命名极具误导性。未来维护者可能错误地传入维度大小而非步长，导致地址计算错误和越界访问。
- **触发条件**: 当前代码功能正确，仅在后续维护/修改时存在引入 bug 的风险。
- **修复建议**: 重命名参数为 stride 语义：
  ```cpp
  uint64_t GetOffset(uint64_t batchStride, uint64_t channelStride, uint64_t lineStride)
  ```
- **测试方案**: 代码审查确认所有调用点传参正确。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 22 行 | Mask值错误/Kernel逻辑 | 严重 | MASK_VAL=0.0f 应为负无穷，导致 mask 位置获得非零 softmax 概率 |
| 2 | 第 68-71 行 | 内存访问/潜在越界 | 中等 | sharedBuffer 扩容判断条件中元素数与字节数单位混用 |
| 3 | 第 46-49 行 | 可维护性 | 低 | GetOffset 参数命名为维度名但实际语义为 stride |
