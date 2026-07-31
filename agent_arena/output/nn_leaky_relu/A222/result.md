# LeakyRelu Tiling (Arch35) 代码审查报告

## Bug 1: FP16 分支未设置 dType 变量

- **位置**: 第 95-99 行
- **类型**: 逻辑错误 / Tiling参数缺失
- **严重程度**: 高
- **描述**: 当 `outputDtype == ge::DT_FLOAT16` 时，代码未对成员变量 `dType` 进行赋值。`dType` 默认初始化为 0，后续在 `SetTilingData()` 中通过 `GET_TPL_TILING_KEY(schMode, dType)` 计算 tiling key，将导致生成错误的 tiling key，可能选择了错误的 kernel 实现。
- **触发条件**: 输入数据类型为 FP16 时触发。
- **测试方案**: 构造 FP16 输入的 LeakyRelu 算子，检查生成的 tiling key 是否正确匹配 FP16 对应的 kernel 模板。对比 BF16 分支（显式设置 `dType = TPL_FP32`），验证 FP16 分支是否需要相同设置。

## Bug 2: BF16 分支错误使用 FP16 的 DAG 模板

- **位置**: 第 103 行
- **类型**: 类型映射错误
- **严重程度**: 高
- **描述**: BF16 分支使用了 `LeakyReluCastDag<half>::OpDag` 进行 tiling 计算。`half` 对应的是 FP16 类型，而非 BF16 类型（应为 `bfloat16_t` 或类似类型）。BF16 和 FP16 虽然都是 16-bit 浮点数，但数据排布和计算语义不同，使用 FP16 的 DAG 模板处理 BF16 数据会导致 tiling 参数（如数据对齐、块大小等）计算错误或运行时计算结果错误。
- **触发条件**: 输入数据类型为 BF16 时触发。
- **测试方案**: 构造 BF16 输入的 LeakyRelu 算子，验证 tiling 结果是否正确；对比使用正确 BF16 DAG 模板的结果，检查数据块切分和对齐是否一致。

## Bug 3: negativeSlope 默认值不合理

- **位置**: 第 93 行
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: 当 `scaleValueAttr` 为 nullptr（属性未设置）时，`negativeSlope` 默认设为 `0.0`。这使得 LeakyRelu 退化为普通 ReLU（负半轴输出为 0）。虽然从数学上不算错误，但 LeakyRelu 的标准默认斜率通常为 `0.01`。如果框架层未传递该属性，算子行为将与用户预期不一致。
- **触发条件**: 调用 LeakyRelu 算子时未设置 negative_slope 属性。
- **测试方案**: 不设置 negative_slope 属性调用算子，验证输出是否符合 LeakyRelu 语义（负半轴斜率应为 0.01 而非 0）。与 PyTorch 等框架的默认行为对比。

## Bug 4: 结构体名称拼写错误 "LeakrReluCompileInfo"

- **位置**: 第 129、138、148 行
- **类型**: 拼写错误（潜在接口不一致）
- **严重程度**: 低
- **描述**: `LeakrReluCompileInfo` 明显是 `LeakyReluCompileInfo` 的拼写错误（缺少 'y'）。虽然如果头文件中也使用相同拼写则可以编译通过，但这种不规范命名会导致维护困难，且如果其他模块使用正确拼写则会出现链接错误。
- **触发条件**: 当其他模块或新代码使用正确拼写 `LeakyReluCompileInfo` 时会导致编译失败。
- **测试方案**: 全局搜索该结构体的定义和使用，确认命名是否一致；尝试使用正确拼写编译，验证是否存在不一致。

## Bug 5: compileInfo 获取后未使用

- **位置**: 第 129-130 行
- **类型**: 逻辑错误 / 冗余代码
- **严重程度**: 低
- **描述**: 在 `TilingForLeakyRelu` 函数中，获取了 `compileInfo` 指针并做了空检查，但之后从未使用该变量。`RunTiling()` 中也未引用编译信息（如 `coreNum`、`ubSize`）。这意味着 `TilingPrepareForLeakyRelu` 中准备的平台信息（核数、UB 大小）在 tiling 计算中可能未被正确传递。
- **触发条件**: 所有场景。若 ElewiseBaseTiling 内部通过 context 获取 compileInfo 则不影响；否则 tiling 计算缺少关键硬件参数。
- **测试方案**: 检查 ElewiseBaseTiling 内部是否自行获取 compileInfo；在不同核数/UB 大小的平台上运行，验证 tiling 结果是否正确适配硬件。

---

# 汇总表

| 编号 | 位置(行号) | 类型 | 严重程度 | 简要描述 |
|------|-----------|------|---------|---------|
| 1 | 95-99 | Tiling参数缺失 | 高 | FP16 分支未设置 dType，导致 tiling key 错误 |
| 2 | 103 | 类型映射错误 | 高 | BF16 分支错误使用 `half`(FP16) 的 DAG 模板 |
| 3 | 93 | 逻辑错误 | 中 | negativeSlope 默认值 0.0 与 LeakyRelu 标准默认值 0.01 不一致 |
| 4 | 129/138/148 | 拼写错误 | 低 | `LeakrReluCompileInfo` 缺少字母 'y' |
| 5 | 129-130 | 逻辑错误/冗余 | 低 | compileInfo 获取后未使用，平台参数可能未参与 tiling |
