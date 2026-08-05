# LeakyRelu Tiling Arch35 代码审查报告

## Bug 列表

### Bug 1: BF16 类型使用了错误的 DAG 模板参数

- **位置**: 第 103 行
- **类型**: 类型映射错误
- **严重程度**: 严重 (High)
- **描述**: 当 `outputDtype == ge::DT_BF16` 时，代码使用了 `LeakyReluCastDag<half>::OpDag` 进行 tiling。`half` 对应的是 FP16 类型，而非 BF16 类型。BF16 应该使用 `LeakyReluCastDag<bfloat16_t>::OpDag`（或项目中对应的 BF16 类型别名）。使用错误的类型模板会导致 tiling 计算中数据大小、对齐要求等参数错误，最终导致计算结果异常或内存越界。
- **触发条件**: 输入数据类型为 BF16 时触发。
- **测试方案**: 构造 BF16 类型的输入 tensor 执行 LeakyRelu 算子，对比输出与预期结果（使用 FP32 参考实现），验证数值正确性。

---

### Bug 2: FP16 分支未设置 dType 值

- **位置**: 第 95-99 行
- **类型**: Tiling 参数错误
- **严重程度**: 严重 (High)
- **描述**: 在 `outputDtype == ge::DT_FLOAT16` 的分支中，成员变量 `dType` 未被赋值，保持默认值 0。后续第 68 行通过 `GET_TPL_TILING_KEY(schMode, dType)` 生成 tiling key 时，dType=0 可能不对应正确的 FP16 tiling key，导致 kernel 选择错误或运行时行为异常。应在该分支中添加 `dType = static_cast<uint64_t>(TPL_FP16);`（或对应的枚举值）。
- **触发条件**: 输入数据类型为 FP16 时触发。
- **测试方案**: 使用 FP16 输入执行算子，检查生成的 tiling key 是否正确匹配 FP16 对应的 kernel 实现；验证端到端计算结果的正确性。

---

### Bug 3: UB Size 被错误地乘以 2

- **位置**: 第 145 行
- **类型**: Tiling 参数错误
- **严重程度**: 中等 (Medium)
- **描述**: `compileInfoPtr->ubSize = compileInfoPtr->ubSize * 2;` 将平台返回的 UB 大小翻倍。在 Ascend 架构中，`GetCoreMemSize` 已经返回了实际可用的 UB 大小，人为翻倍会导致 tiling 计算时认为可用内存大于实际值，可能造成 UB 内存越界访问，导致计算错误或硬件异常。除非有明确的架构设计原因（如 arch35 特定的双 bank 合并场景且 API 返回单 bank 大小），否则这是一个错误。
- **触发条件**: 所有情况下都会触发，当 tiling 计算使用了接近 UB 上限的内存分配时表现为实际错误。
- **测试方案**: 使用大 shape 输入（使 tiling 分块接近 UB 容量上限）执行算子，观察是否出现内存越界错误；对比去除 `*2` 后的正确性。

---

### Bug 4: 结构体命名拼写错误 "LeakrReluCompileInfo"

- **位置**: 第 129、138、149 行
- **类型**: 命名/拼写错误（潜在维护风险）
- **严重程度**: 低 (Low)
- **描述**: `LeakrReluCompileInfo` 明显是 `LeakyReluCompileInfo` 的拼写错误（缺少字母 'y'）。虽然如果头文件中定义了同样拼写错误的名称则编译不会报错，但这属于代码质量问题，会造成维护困难和与其他模块集成时的命名不一致。
- **触发条件**: 代码维护或重构时容易引发混淆。
- **测试方案**: 全局搜索该拼写，确认是否与头文件定义一致；若不一致则编译阶段即可发现。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 103 行 | 类型映射错误 | 严重 | BF16 分支错误使用 `half` 模板参数，应使用 bfloat16 类型 |
| 2 | 第 95-99 行 | Tiling 参数错误 | 严重 | FP16 分支未设置 `dType`，导致 tiling key 生成错误 |
| 3 | 第 145 行 | Tiling 参数错误 | 中等 | UB Size 被错误乘以 2，可能导致内存越界 |
| 4 | 第 129/138/149 行 | 命名拼写错误 | 低 | `LeakrReluCompileInfo` 拼写错误，缺少字母 'y' |
