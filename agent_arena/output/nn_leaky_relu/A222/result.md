# LeakyRelu Tiling (arch35) Code Review Report

## Bug List

### Bug 1: BF16 分支使用了错误的 DAG 模板类型

- **位置**: 第 103 行
- **类型**: 类型映射错误 (Type Mapping Bug)
- **严重程度**: 高 (High)
- **描述**: 当 `outputDtype == ge::DT_BF16` 时，代码使用了 `LeakyReluCastDag<half>::OpDag` 进行 tiling 计算。`half` 对应的是 FP16 类型，而非 BF16 类型。BF16 应该使用 `LeakyReluCastDag<bfloat16_t>::OpDag`（或对应的 bf16 模板参数）。使用错误的模板类型会导致 UB buffer 大小计算错误（half 和 bfloat16 虽然都是 2 字节，但 cast 到 fp32 的中间 buffer 布局和对齐要求可能不同），以及运行时数据处理逻辑不匹配。
- **触发条件**: 输入数据类型为 BF16 时触发。
- **测试方案**: 构造 BF16 类型输入 tensor，执行 LeakyRelu 算子，对比输出结果与预期值（CPU 参考实现），验证精度是否正确。

### Bug 2: FP16 分支未设置 dType 变量

- **位置**: 第 95-99 行
- **类型**: Tiling 参数错误 (Tiling Parameter Bug)
- **严重程度**: 高 (High)
- **描述**: 在 `outputDtype == ge::DT_FLOAT16` 分支中，没有对成员变量 `dType` 进行赋值。`dType` 初始化为 0，后续在 `SetTilingData()` 中通过 `GET_TPL_TILING_KEY(schMode, dType)` 生成 tiling key。如果 `dType=0` 不对应有效的 FP16 类型枚举值（如应为 `TPL_FP16`），则会生成错误的 tiling key，导致 kernel 选择错误或运行时异常。
- **触发条件**: 输入数据类型为 FP16 时触发。
- **测试方案**: 构造 FP16 类型输入，检查生成的 tiling key 是否与预期 FP16 对应的 key 匹配；运行算子验证是否能正确调度到对应 kernel。

### Bug 3: UB Size 未传递给 ElewiseBaseTiling

- **位置**: 第 79 行及 `TilingForLeakyRelu` 函数（第 129 行）
- **类型**: UB Size 参数缺失 (UB Size Bug)
- **严重程度**: 中 (Medium)
- **描述**: 在 `TilingPrepareForLeakyRelu`（第 144 行）中获取了 `compileInfoPtr->ubSize`，但在 `RunTiling()` 中创建 `ElewiseBaseTiling` 对象时，并未将 `compileInfo` 中的 `ubSize` 信息显式传入或使用。虽然 `ElewiseBaseTiling` 可能通过 context 内部获取，但 `TilingForLeakyRelu` 中获取了 `compileInfo`（第 129 行）却没有将其传递给 `LeakyReluTiling` 或 `ElewiseBaseTiling`，存在 UB size 信息丢失的风险。如果 `ElewiseBaseTiling` 依赖 compile info 中的 ubSize 来计算分块，则可能使用默认值导致 UB 溢出。
- **触发条件**: 在 UB 较小的硬件平台上运行大 tensor 时，若分块计算使用了错误的 UB size，可能导致内存越界。
- **测试方案**: 在不同 UB size 的平台上运行，观察是否有 UB 溢出报错；通过日志确认 tiling 使用的 UB size 值是否与硬件实际一致。

### Bug 4: compileInfo 获取后未使用（逻辑冗余/潜在遗漏）

- **位置**: 第 129 行
- **类型**: 逻辑错误 (Logic Bug)
- **严重程度**: 中 (Medium)
- **描述**: `TilingForLeakyRelu` 函数中通过 `reinterpret_cast` 获取了 `compileInfo` 指针并做了空指针检查，但后续完全没有使用该变量。`compileInfo` 包含了 `coreNum` 和 `ubSize` 等关键信息，这些信息应在 tiling 计算中使用（例如传递给 `ElewiseBaseTiling` 或用于 blockDim 计算）。未使用意味着 tiling 计算可能缺少平台相关的关键参数。
- **触发条件**: 所有调用路径均受影响。
- **测试方案**: 检查 `ElewiseBaseTiling` 内部是否能独立从 context 获取 ubSize/coreNum；若不能，则 tiling 结果会偏离预期，可通过对比不同核数平台的 tiling 结果验证。

### Bug 5: 结构体名称疑似拼写错误 `LeakrReluCompileInfo`

- **位置**: 第 129、138、148 行
- **类型**: 命名/拼写错误 (Naming Bug)
- **严重程度**: 低 (Low) - 编译期可发现
- **描述**: `LeakrReluCompileInfo` 疑似为 `LeakyReluCompileInfo` 的拼写错误（缺少字母 'y'）。如果头文件中定义的结构体名也是 `LeakrReluCompileInfo`，则功能正常但影响代码可读性和维护性；如果头文件中为正确拼写 `LeakyReluCompileInfo`，则会导致编译失败。
- **触发条件**: 编译时触发（若头文件定义与此不一致）。
- **测试方案**: 编译验证；代码审查确认头文件中对应结构体的命名。

---

## Bug 汇总表

| 编号 | 位置 (行号) | 类型 | 严重程度 | 简述 |
|------|-------------|------|----------|------|
| 1 | 103 | 类型映射错误 | 高 | BF16 分支错误使用 `half` 模板，应为 bf16 对应类型 |
| 2 | 95-99 | Tiling 参数错误 | 高 | FP16 分支未设置 `dType`，导致 tiling key 错误 |
| 3 | 79, 129 | UB Size 缺失 | 中 | compileInfo 中的 ubSize 未传递给 tiling 计算逻辑 |
| 4 | 129 | 逻辑错误 | 中 | compileInfo 获取后未使用，平台参数可能丢失 |
| 5 | 129, 138, 148 | 拼写错误 | 低 | `LeakrReluCompileInfo` 疑似缺少字母 'y' |
