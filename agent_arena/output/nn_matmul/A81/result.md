# MatMul V3 Base Tiling 代码审查报告

## Bug 列表

### Bug 1: AOE TilingEnable 解析跳过百位数字

- **位置**: 第 133-139 行, `CheckAoeTilingEnable` 函数
- **类型**: Tiling参数解析逻辑错误
- **严重程度**: 高
- **描述**: 代码从十位 (`aoeTilingEnable / 10U`) 直接跳到千位 (`aoeTilingEnable / 1000U`)，完全跳过了百位的提取。百位数字被静默忽略，导致 AOE tiling 参数的百位信息丢失，无法正确配置对应的 tiling 功能。
- **触发条件**: 当 AOE 传入的 `aoeTilingEnable` 参数的百位非零时（例如 `aoeTilingEnable = 1230`），百位 `2` 对应的功能将不会被解析和使能。
- **测试方案**: 构造 `aoeTilingEnable` 值使百位非零（如 100、200），验证对应功能位是否被正确解析并生效。

---

### Bug 2: CheckMMTilingDataIsVaild 中 debug 字符串错误

- **位置**: 第 2659 行和第 2661 行, `CheckMMTilingDataIsVaild` 函数
- **类型**: Debug代码错误（复制粘贴错误）
- **严重程度**: 低
- **描述**: 
  - 第 2659 行: `CheckNumberIsValid(runInfo_.stepM, args_.opName, "runInfo_.baseK")` — 实际校验的是 `stepM`，但日志字符串写成了 `"runInfo_.baseK"`。
  - 第 2661 行: `CheckNumberIsValid(runInfo_.stepKa, args_.opName, "runInfo_.baseK")` — 实际校验的是 `stepKa`，但日志字符串写成了 `"runInfo_.baseK"`。
- **触发条件**: 当 `stepM` 或 `stepKa` 校验失败时，输出的错误日志会误导开发者以为 `baseK` 出了问题。
- **测试方案**: 强制让 `stepM` 或 `stepKa` 为非法值，检查错误日志输出是否准确反映出错的字段名称。

---

### Bug 3: UB_SIZE 常量值疑似错误

- **位置**: 第 84 行
- **类型**: 常量定义错误
- **严重程度**: 中
- **描述**: `constexpr uint64_t UB_SIZE = 196352 * 2;` 计算结果为 392704。Ascend NPU 标准 UB 大小为 192KB = 196608 字节，双倍为 393216。而 196352 = 196608 - 256，差了 256 字节。该常量在第 1226 行用于 `compileInfo_.ubSize == UB_SIZE` 的平台匹配判断，如果平台实际 UB 为标准 192KB*2=393216，则此比较永远为 false，导致 mata 冲突优化路径永远不会触发。
- **触发条件**: 在 UB 大小为标准 393216 字节的平台上执行 mata 冲突场景（dValue 为 16384 倍数，nValue >= 7168，fp16/bf16，24核）。
- **测试方案**: 在标准 24 核平台上构造满足 mata 冲突条件的 shape（如 N=16384, nValue=7168, fp16），验证是否进入 `baseN=96, baseD=512` 的优化分支。

---

### Bug 4: TilingKey 日志格式说明符与类型不匹配

- **位置**: 第 2725 行, `DoTilingKey` 函数
- **类型**: Debug代码错误
- **严重程度**: 低
- **描述**: `OP_LOGI(args_.opName, "Tiling Key is 0x%x", tilingKey_);` 中 `tilingKey_` 为 `uint64_t` 类型，但格式说明符 `%x` 仅打印 32 位。在 64 位系统上，高 32 位将被截断，导致日志中显示的 tiling key 值不完整、不正确。
- **触发条件**: 当 `tilingKey_` 的值超过 `0xFFFFFFFF`（即高 32 位非零）时，日志打印结果错误。根据 `GET_TPL_TILING_KEY` 的生成逻辑，tiling key 通常是大数值，极有可能超过 32 位。
- **测试方案**: 检查任意正常 tiling 流程的日志输出，对比实际 `tilingKey_` 值与日志打印值是否一致。应改为 `%lu` 或 `PRIx64`。

---

### Bug 5: 有符号/无符号类型混合比较

- **位置**: 第 2470 行, `SupportMultiSplitK` 函数
- **类型**: 类型安全错误
- **严重程度**: 低
- **描述**: `mCnt * nCnt < static_cast<int64_t>(compileInfo_.aicNum) / NUMBER_TWO` 将 `uint64_t` 类型的 `compileInfo_.aicNum` 转为 `int64_t` 再与 `uint64_t` 的 `mCnt * nCnt` 比较。虽然在当前场景下不会溢出，但类型转换方向错误，应使用 `uint64_t` 保持一致性，避免潜在的有符号比较陷阱。
- **触发条件**: 理论上当 `aicNum` 非常大（超过 INT64_MAX）时会出现问题，实际硬件不会触发，但属于不规范代码。
- **测试方案**: 静态代码分析工具检测混合类型比较告警。

---

### Bug 6: 函数名拼写错误 "Vaild" 应为 "Valid"

- **位置**: 第 2648 行, 函数声明 `CheckMMTilingDataIsVaild`
- **类型**: 命名错误（拼写）
- **严重程度**: 低
- **描述**: 函数名 `CheckMMTilingDataIsVaild` 中 "Vaild" 是 "Valid" 的拼写错误。虽不影响运行时功能，但影响代码可维护性和可搜索性。
- **触发条件**: 无功能影响，但在代码搜索、重构时可能造成困惑。
- **测试方案**: 全局搜索确认所有调用处拼写一致后统一修正。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | L133-139 | Tiling参数解析 | 高 | AOE tilingEnable 百位数字未解析，功能丢失 |
| 2 | L2659,2661 | Debug代码 | 低 | 校验失败日志字符串与实际字段不匹配 |
| 3 | L84 | 常量定义 | 中 | UB_SIZE=196352*2 与标准192KB*2=393216不符，平台匹配可能失败 |
| 4 | L2725 | Debug代码 | 低 | uint64_t 使用 %x 格式符，高32位被截断 |
| 5 | L2470 | 类型安全 | 低 | uint64_t 与 int64_t 混合比较 |
| 6 | L2648 | 命名错误 | 低 | 函数名 "Vaild" 应为 "Valid" |
