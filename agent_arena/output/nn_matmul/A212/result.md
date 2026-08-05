# MatMul V3 Base Tiling 代码审查报告

## Bug 列表

### Bug 1: L1 Size 计算方向错误 (加法应为减法)

- **位置**: 第 581 行, `MatmulV3BaseTiling::CalL1Tiling()`
- **类型**: Tiling 参数计算错误
- **严重程度**: 高
- **描述**: `uint64_t totalL1Size = compileInfo_.l1Size + 256;` 代码注释明确说明 "256B为预留给rpc使用，单算子不涉及"，即这256字节是需要预留(排除)的空间。但实际代码却将256B **加到** L1总大小上，导致计算出的可用L1空间比实际硬件更大。后续基于此计算的 `depthA1` 和 `depthB1` 可能超出L1物理容量，导致数据被覆盖或硬件异常。
- **触发条件**: 所有走 `CalL1Tiling()` 路径且 `supportL0c2out == true` 的场景，当L1使用率接近满载时会触发实际溢出。
- **测试方案**: 构造使L1接近满载的用例 (大baseM/baseN、大depthA1/depthB1)，对比 `totalL1Size` 与实际L1 size，验证depthA1*baseM*baseK*dtype + depthB1*baseN*baseK*dtype 是否超出物理L1。

---

### Bug 2: aoeTilingEnable 百位解析缺失

- **位置**: 第 133-144 行, `MatmulV3BaseTiling::CheckAoeTilingEnable()`
- **类型**: Tiling 参数解析遗漏
- **严重程度**: 中
- **描述**: 代码从 `aoeTilingEnable` 数值中按位提取配置: 个位(line 122)、十位(line 133)、千位(line 139)、万位(line 146)。但**百位**被完全跳过，没有任何提取和校验逻辑。如果AOE产生的 tilingEnable 中百位有有效值，将被静默忽略，导致 tiling 策略不完整。
- **触发条件**: AOE 调优产生的 `aoeTilingEnable` 值百位非零时 (例如 `x1y00` 形式的值)，百位配置无法生效。
- **测试方案**: 设置 `aoeTilingEnable` 为含有百位非零的值 (如 12345)，验证是否有未生效的配置项，对比与预期 tiling 行为的差异。

---

### Bug 3: 310P N轴对齐检查使用错误的 dtype size

- **位置**: 第 835 行, `MatmulV3BaseTiling::CheckDimsAligned310P()`
- **类型**: 对齐检查错误
- **严重程度**: 高
- **描述**: `if (args_.nOriValue * bDtypeSize_ % BLOCK_BYTE_SIZE != 0)` 检查N轴是否32字节对齐时使用了B矩阵的 `bDtypeSize_`。但此处检查的是**输出格式** (`args_.outFormat == ge::FORMAT_ND`) 下输出矩阵N维度的对齐性，应使用输出数据类型大小 `cDtypeSize_`。当输入为fp16 (2B) 但输出为fp32 (4B) 时 (如 fp16*fp16->fp32 的支持组合)，用 bDtypeSize_=2 检查会通过，但实际输出 fp32 数据的对齐要求是 `nOriValue * 4 % 128 == 0`，更为严格。
- **触发条件**: 310P 平台, ND 输出格式, 输入为 fp16/bf16, 输出为 fp32, 且 N 满足 `N*2%128==0` 但 `N*4%128!=0` (例如 N=48: 48*2=96, 96%128!=0 也不通过; N=16: 通过)。实际当 bDtype != cDtype 且 N 值恰好处于判断分界时触发。
- **测试方案**: 在310P平台构造 fp16->fp32 输出场景, 设置 N 使得 `N*bDtype%128==0` 但 `N*cDtype%128!=0`, 验证是否产生非对齐访存错误。

---

### Bug 4: 属性类型不一致 (bool vs int64_t)

- **位置**: 第 241 行 (CheckArgs) vs 第 263 行 (GetDtype)
- **类型**: 类型混淆 / 潜在内存越界
- **严重程度**: 中
- **描述**: 在 `CheckArgs()` 中 (第241行)，`OP_IMPL_MODE_ATTR_INDEX` (=3) 位置的属性被校验为 `bool` 类型: `attrs->GetAttrPointer<bool>(OP_IMPL_MODE_ATTR_INDEX)`。但在 `GetDtype()` 中 (第263行)，同一位置被读取为 `int64_t`: `*context.GetAttrs()->GetAttrPointer<int64_t>(OP_IMPL_MODE_ATTR_INDEX)`。如果底层属性存储为 `bool` (1字节)，以 `int64_t` (8字节) 方式读取将读到相邻内存的脏数据，可能导致 `isHf32` 或 `isForceGrpAccForFp32` 被错误设置。
- **触发条件**: 当 `context->GetNodeType()` 为 "MatMulV3" 且属性数量 >= 4 时，每次都会触发此路径。实际是否产生错误取决于 GetAttrPointer 的内部实现对类型的处理。
- **测试方案**: 构造 MatMulV3 算子, 设置第4个属性为 `op_impl_mode=0x40`，验证 `isHf32` 是否被正确识别; 打印 GetAttrPointer<bool> 和 GetAttrPointer<int64_t> 在同一index的返回值对比。

---

### Bug 5: N轴 singleCoreN 对齐使用了 A 矩阵的 dtypeSize

- **位置**: 第 2402 行, `MatmulV3BaseTiling::DoSingleCoreSplitKTiling()`
- **类型**: 对齐参数错误
- **严重程度**: 低-中
- **描述**: `uint64_t nAlignLength = ALIGN_INNER / aDtypeSize_;` 计算N方向的对齐粒度时使用了A矩阵的dtype大小 (`aDtypeSize_`)。但N维度属于B矩阵的维度，正确应使用 `bDtypeSize_`。当 A/B dtype 不同时 (如 A 为 fp32, B 为 fp16)，会导致N方向对齐粒度计算错误。但在当前支持的dtype组合中 (fp16/fp16, bf16/bf16, fp32/fp32), A和B的dtype始终相同，因此目前不会产生实际错误。
- **触发条件**: 仅在未来扩展支持 A/B 不同数据类型时 (如混合精度输入) 会触发。当前dtype支持列表中 A/B 始终同类型。
- **测试方案**: 若扩展混合精度支持，设置 A=fp32, B=fp16，验证 singleCoreN 的对齐是否正确 (应为 256/2=128 而非 256/4=64)。

---

### Bug 6: IsGmToL1ByShape 中条件逻辑矛盾

- **位置**: 第 2314-2318 行, `MatmulV3BaseTiling::IsGmToL1ByShape()`
- **类型**: 逻辑错误 (死代码)
- **严重程度**: 低
- **描述**: 第2314行先检查 `!args_.hasBias`，只有无bias时才进入分支。但第2316行又检查 `if (runInfo_.stepM == STEP_NUM_3 && args_.hasBias)` 设置 L1 bias size。由于外层已排除 hasBias，内层的 `args_.hasBias` 永远为 false，`shareL1Size = L1_BIAS_SIZE` 永远不会被执行。这意味着 GM_TO_L1 模板在有 bias 场景下永远不会正确预留 bias 空间。
- **触发条件**: 由于外层 `!args_.hasBias` 条件，有 bias 时根本不会进入此函数体，所以当前不会产生运行时错误。但如果未来修改外层条件，内层逻辑将失效。
- **测试方案**: 代码走读确认即可。若意图是有bias也能走 GM_TO_L1 路径，需将外层条件修改为允许 hasBias。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | L581 | Tiling参数计算 | 高 | L1可用大小应减去256B预留，代码错误地加了256 |
| 2 | L133-144 | Tiling参数解析 | 中 | aoeTilingEnable百位未解析，配置可能丢失 |
| 3 | L835 | 对齐检查 | 高 | 310P输出N轴对齐检查用bDtypeSize_应为cDtypeSize_ |
| 4 | L241/263 | 类型混淆 | 中 | 同一属性index在CheckArgs中校验为bool，在GetDtype中读取为int64_t |
| 5 | L2402 | 对齐参数 | 低-中 | N方向对齐粒度用aDtypeSize_应为bDtypeSize_ |
| 6 | L2314-2318 | 逻辑矛盾 | 低 | 外层!hasBias使内层hasBias判断永远为false (死代码) |
