# ScaledMaskedSoftmaxV2 Tiling 代码审查报告

## Bug 列表

### Bug 1: 对齐常量 AlignedBytes 值错误

- **位置**: 第 33 行 `constexpr uint64_t AlignedBytes = 31;`
- **类型**: 对齐常量错误
- **严重程度**: 高
- **描述**: Ascend NPU 上数据对齐要求是 32 字节（一个 block 大小）。`AlignedBytes` 应为 32 而非 31。当前值 31 被用作除数计算 `alignedXBlock = AlignedBytes / xDtypeSize`（第216行）和 `alignedMaskBlock = AlignedBytes / BOOL_SIZE`（第222行），导致计算出的对齐块数不正确（例如 31/4=7 而非 32/4=8，31/2=15 而非 32/2=16），从而产生错误的 padding 和地址不对齐问题，可能导致 DMA 传输异常或计算结果错误。
- **触发条件**: 所有情况下均会触发，width 不是 7/15/31 倍数时 padding 计算错误。
- **测试方案**: 使用 width=8 的 FP32 输入，预期 paddingNum=0（8 已对齐到 8），但当前会计算为 alignedXBlock=7, xLeft=8%7=1, paddingNum=6，导致错误 padding。

---

### Bug 2: DTYPE_DIGIT 常量值为 1 导致 TilingKey 无法区分数据类型

- **位置**: 第 40 行 `constexpr uint64_t DTYPE_DIGIT = 1;`，第 316 行 `uint64_t tilingKey = dtypeKey * DTYPE_DIGIT;`
- **类型**: Tiling 参数错误
- **严重程度**: 中
- **描述**: `DTYPE_DIGIT` 的命名暗示它应是数位乘数（如 10、100），用于将 dtype 编码到 tilingKey 的特定数位上。但其值为 1，使得 `tilingKey = dtypeKey * 1 = dtypeKey`。当前 FP32 的 tilingKey 为 0，这可能导致 kernel 侧无法通过 tilingKey 正确派发到不同数据类型的实现分支（tilingKey=0 可能被视为无效或默认值）。如果后续需要在 tilingKey 中编码更多信息（如其他模式位），乘以 1 无法预留数位空间。
- **触发条件**: 使用 FP32 数据类型时 tilingKey=0；若 kernel 侧依赖 tilingKey 的数位编码逻辑则所有类型都受影响。
- **测试方案**: 分别用 FP32/FP16/BF16 输入运行，检查 kernel 是否被正确调度到对应 dtype 模板实例。

---

### Bug 3: Init() 返回值未检查导致未初始化成员被使用

- **位置**: 第 342-344 行 `TilingScaledMaskedSoftmaxV2` 函数
- **类型**: 初始化逻辑错误
- **严重程度**: 高
- **描述**: `tilingObj.Init()` 的返回值被忽略。如果 `Init()` 失败（返回非 `GRAPH_SUCCESS`），程序仍会继续调用 `DoTiling()`，此时 `batch`、`channel`、`height`、`width`、`xDtypeSize` 等成员变量未被正确初始化（为随机值或0），将导致后续计算产生未定义行为，包括除零崩溃（`xDtypeSize=0` 时第216行除零）或内存分配异常。
- **触发条件**: 输入 shape 非 4 维、dtype 不支持、平台信息获取失败等任何 Init 子步骤失败时触发。
- **测试方案**: 传入 3 维 shape 的输入 tensor，验证是否正确返回 GRAPH_FAILED 而非崩溃。

---

### Bug 4: tailCore 行数为 0 时 uint64_t 下溢

- **位置**: 第 279-281 行 `iterTailCore` 和 `lineLastTailIter` 的计算
- **类型**: Tiling 参数计算溢出
- **严重程度**: 中
- **描述**: 当 `totalLine` 能被 `coreNum` 整除时，`headCoreNum=0`，所有 core 都是 tailCore，逻辑正常。但当 `totalLine < coreNum` 时，`lineTailCore = totalLine/coreNum = 0`，`iterTailCore = CeilDiv(0, availableLinePerIter) = 0`。之后第 281 行计算 `lineLastTailIter = 0 - (0-1) * lineTailIter`，由于 `iterTailCore-1` 是 uint64_t 下溢为极大值（约 1.8e19），导致乘法溢出和减法溢出，产生错误的 tiling 参数。
- **触发条件**: `totalLine < coreNum`，例如 batch=1, channel=1, height=1 且 coreNum > 1。
- **测试方案**: 设置 batch=1, channel=1, height=1, width=128，在多核平台（如 coreNum=32）上运行，检查 tiling 参数是否合理。

---

### Bug 5: SetSoftmaxTiling 中 SOFTMAX_BUF_SIZE 未适配 910_95 平台

- **位置**: 第 301 行 `if (size > SOFTMAX_BUF_SIZE)`
- **类型**: 平台适配错误
- **严重程度**: 中
- **描述**: 在 `SetUbSplitInfo()` 中已正确区分 910_95 平台使用 `SOFTMAX_BUF_SIZE_D`(64KB) vs 其他平台使用 `SOFTMAX_BUF_SIZE`(32KB) 来计算可用行数。但在 `SetSoftmaxTiling()` 中，第 301 行硬编码使用 `SOFTMAX_BUF_SIZE`(32KB) 作为上限。对于 910_95 平台，softmax tmp buffer 实际可使用 64KB，但这里将其截断为 32KB，导致 SoftMaxTilingFunc 生成的 tiling 参数与实际分配的 buffer 大小不匹配，可能导致性能下降或功能异常。
- **触发条件**: 在 ASCEND910_95 平台上运行时触发。
- **测试方案**: 在 910_95 平台上运行大 width 场景，对比 softmax tiling 参数是否与 UB 分配一致。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第33行 | 对齐常量错误 | 高 | `AlignedBytes=31` 应为 32，导致所有对齐计算错误 |
| 2 | 第40/316行 | Tiling参数错误 | 中 | `DTYPE_DIGIT=1` 使 tilingKey 编码失去数位意义，FP32 key=0 |
| 3 | 第342-344行 | 初始化逻辑错误 | 高 | `Init()` 返回值未检查，失败后继续执行导致未定义行为 |
| 4 | 第279-281行 | 整数溢出 | 中 | `lineTailCore=0` 时 uint64_t 下溢产生极大值 |
| 5 | 第301行 | 平台适配错误 | 中 | 910_95 平台 softmax buffer 上限未使用 64KB 常量 |
