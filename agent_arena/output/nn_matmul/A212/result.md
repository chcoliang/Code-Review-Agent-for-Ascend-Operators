# MatMul V3 Base Tiling 代码审查报告

## Bug 1: aoeTilingEnable 解析跳过百位数字

- **位置**: 第 139 行
- **类型**: Tiling 参数解析逻辑缺陷
- **严重程度**: 高
- **描述**: `CheckAoeTilingEnable` 函数在解析 `aoeTilingEnable` 的十进制编码时，从十位 (`/10U % 10U`) 直接跳到千位 (`/1000U % 10U`)，完全跳过了百位 (`/100U % 10U`) 的解析。如果百位包含有效的 tiling 配置信息，该信息将被丢失。
- **触发条件**: 当 AOE 调优生成的 `aoeTilingEnable` 值的百位非零时（如 `tilingEnable >= 100`），百位参数被忽略。
- **测试方案**: 设置 `aoeTilingEnable = 1234`，验证百位数字 `2` 是否被正确解析并应用到对应的 tiling 策略中。

## Bug 2: 输出 N 维度对齐检查使用了错误的 dtype size

- **位置**: 第 834 行
- **类型**: 对齐检查错误
- **严重程度**: 高
- **描述**: `CheckDimsAligned310P` 中对输出张量 N 维度的 32 字节对齐检查使用了 `bDtypeSize_`（B 矩阵输入数据类型大小），但输出张量的数据类型是 `cType`，应使用 `cDtypeSize_`。当输入为 fp16 而输出为 fp32 时（`bDtypeSize_=2, cDtypeSize_=4`），对齐判断会出错。
- **触发条件**: 在 310P 平台上，输入 B 为 fp16/bf16，输出为 fp32，且 `nOriValue` 满足 `nOriValue * 2 % 128 == 0` 但 `nOriValue * 4 % 128 != 0` 时（如 N=48），本应报错但会错误放行。
- **测试方案**: 在 310P 平台设置 `aType=DT_FLOAT16, bType=DT_FLOAT16, cType=DT_FLOAT, nOriValue=48, outFormat=FORMAT_ND`，验证是否正确报出对齐错误。

## Bug 3: N 轴 singleCore 对齐计算使用了 A 矩阵的 dtype size

- **位置**: 第 2402 行
- **类型**: Tiling 参数计算错误
- **严重程度**: 中
- **描述**: 在 `DoSingleCoreSplitKTiling` 中，`nAlignLength = ALIGN_INNER / aDtypeSize_` 用于计算 N 方向的 singleCoreN 对齐粒度。但 N 是 B 矩阵的维度，应使用 `bDtypeSize_`。当 A/B 数据类型不同时会导致 singleCoreN 对齐粒度错误。
- **触发条件**: 当 `aDtypeSize_ != bDtypeSize_` 时触发（当前 MatMulV3 支持的 dtype 组合中 A/B 类型相同，故暂未实际触发，但代码逻辑存在隐患）。
- **测试方案**: 若未来扩展支持混合精度（如 A=fp32, B=fp16），设置对应数据类型并触发单核切K路径，检查 singleCoreN 的对齐是否正确。

## Bug 4: dtype_bias 字段被错误赋值为数据类型大小

- **位置**: 第 1177 行
- **类型**: 参数赋值错误
- **严重程度**: 中
- **描述**: `InitRunParams` 中 `runParams.dtype_bias = ge::GetSizeByDataType(args.biasType)` 将 bias 的**字节大小**赋给了 `dtype_bias` 字段。对比其他 dtype 字段（`dtype_a`, `dtype_b`, `dtype_out`）均赋值为 DataType 枚举值，且第 1185 行已有 `runParams.bias_dtype = args.biasType` 正确赋值。`dtype_bias` 应为 `args.biasType`。
- **触发条件**: 当使用 V2 tiling 路径（`GetV2Tiling`）且有 bias 时，下游逻辑若依赖 `dtype_bias` 作为数据类型枚举使用，会得到错误值（如 fp32 的 size=4 被误解为 DT_INT8=4）。
- **测试方案**: 设置 `hasBias=true, biasType=DT_FLOAT(size=4)`，执行 V2 tiling 路径，验证 `runParams.dtype_bias` 是否与预期的 DataType 枚举值匹配。

## Bug 5: CheckMMTilingDataIsVaild 中日志字段名复制粘贴错误

- **位置**: 第 2659 行、第 2661 行
- **类型**: 日志/调试信息错误
- **严重程度**: 低
- **描述**: 第 2659 行检查 `runInfo_.stepM` 但错误信息写的是 `"runInfo_.baseK"`；第 2661 行检查 `runInfo_.stepKa` 但错误信息也写的是 `"runInfo_.baseK"`。这会导致校验失败时打印的错误日志信息指向错误的字段，误导定位问题。
- **触发条件**: 当 `stepM` 或 `stepKa` 为无效值（如 0 或超大值）触发校验失败时，日志输出错误字段名。
- **测试方案**: 构造 `runInfo_.stepM = 0` 的场景，检查报错日志是否正确输出 "runInfo_.stepM" 而非 "runInfo_.baseK"。

## Bug 6: VNCHW_UP_THRES 常量值疑似定义错误

- **位置**: 第 65 行
- **类型**: 常量定义疑似错误
- **严重程度**: 低
- **描述**: `VNCHW_UP_THRES = 72368` 不是任何常见对齐粒度的整数倍（非 16/32/64/128/256/512 对齐），也不是典型 NPU 硬件参数的组合值。相近的合理值包括 `73728 = 9*8192` 或 `72704 = 71*1024`。此常量用于第 1868 行判断外轴大小是否满足 vnchw 转换条件，错误的阈值可能导致部分 shape 走入次优路径。
- **触发条件**: 当矩阵外轴值在 72368 附近时（如 72368~73728 之间），判断结果可能与预期不符。
- **测试方案**: 对比 outerSize=72368 和 outerSize=72367 的 tiling 路径选择是否符合性能预期，验证阈值的合理性。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 139 | Tiling参数解析 | 高 | aoeTilingEnable 跳过百位解析 |
| 2 | 834 | 对齐检查 | 高 | 输出N维度对齐用了bDtypeSize_而非cDtypeSize_ |
| 3 | 2402 | Tiling参数计算 | 中 | N轴对齐粒度误用aDtypeSize_ |
| 4 | 1177 | 参数赋值 | 中 | dtype_bias被赋值为size而非DataType枚举 |
| 5 | 2659/2661 | 日志信息 | 低 | 复制粘贴导致错误字段名 |
| 6 | 65 | 常量定义 | 低 | VNCHW_UP_THRES=72368 非标准对齐值 |
