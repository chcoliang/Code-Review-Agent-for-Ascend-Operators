# aclnn_leaky_relu.cpp 代码审查报告

## Bug 列表

### Bug 1: CheckShape 未校验输出 tensor 与输入 tensor 的 shape 一致性

- **位置**: 第 61-63 行，`CheckShape` 函数
- **类型**: 逻辑缺陷 / 参数校验不完整
- **严重程度**: 高
- **描述**: 函数注释明确写道"输出和输入的shape是否相同"，但实现中仅检查了 `self` 的最大维度数，完全没有校验 `out` 的 shape 是否与 `self` 一致。当用户传入 shape 不匹配的 output tensor 时，后续 `ViewCopy` 会导致内存越界写入或数据错误。
- **触发条件**: 调用 `aclnnLeakyReluGetWorkspaceSize` 时传入的 `out` tensor 的 shape 与 `self` 不一致（如 self 为 [2,3]，out 为 [4,5]）。
- **测试方案**: 构造 self shape=[2,3,4]，out shape=[1,2,3] 的 tensor 对，调用接口，期望返回 `ACLNN_ERR_PARAM_INVALID`，实际会跳过校验进入计算流程。

---

### Bug 2: negativeSlope 使用 ToFloat() 导致 DT_DOUBLE 精度丢失

- **位置**: 第 103 行，`negativeSlope->ToFloat()`
- **类型**: 精度损失
- **严重程度**: 中
- **描述**: `negativeSlope` 标量无条件转换为 `float`（32位）。当输入 tensor 的数据类型为 `DT_DOUBLE`（64位）时，negativeSlope 的高精度值会被截断为 float 精度，导致计算结果与预期不符。应根据输入 tensor 的 dtype 决定使用 `ToFloat()` 还是 `ToDouble()`。
- **触发条件**: 输入 tensor 为 DT_DOUBLE 类型，且 negativeSlope 为需要高精度表示的值（如 0.123456789012345678）。
- **测试方案**: 构造 DT_DOUBLE 类型输入，negativeSlope 设为需超过 float 精度的值（如 1e-10 级别的微小差异），对比 CPU 参考实现的结果，验证精度偏差是否超过 double 容许误差。

---

### Bug 3: 未校验输出 tensor 的数据类型是否在支持列表内

- **位置**: 第 54-58 行，`CheckDtypeValid` 函数
- **类型**: 参数校验不完整
- **严重程度**: 中
- **描述**: `CheckDtypeValid` 仅检查了输入 `self` 的 dtype 是否在支持列表中，但未检查输出 `out` 的 dtype。如果用户传入不支持的输出类型（如 INT32），后续 `Cast` 操作可能失败或产生未定义行为。
- **触发条件**: 调用时 self 为 DT_FLOAT（合法），但 out 为 DT_INT8 或 DT_INT32 等不支持的类型。
- **测试方案**: 构造 self 为 DT_FLOAT、out 为 DT_INT32 的 tensor，调用接口，期望返回错误码但实际会进入 Cast 流程。

---

### Bug 4: 未对 workspaceSize 和 executor 指针参数进行空指针校验

- **位置**: 第 79-80 行，`aclnnLeakyReluGetWorkspaceSize` 函数入口
- **类型**: 空指针解引用风险
- **严重程度**: 中
- **描述**: 函数参数 `workspaceSize` 和 `executor` 是指针类型，但函数入口处未做空指针检查。若调用者传入 nullptr，在第 93 行 `*workspaceSize = 0` 或第 115 行 `*workspaceSize = ...` 处会触发段错误（segfault）。
- **触发条件**: 调用 `aclnnLeakyReluGetWorkspaceSize` 时 `workspaceSize` 或 `executor` 传入 `nullptr`。
- **测试方案**: 分别以 workspaceSize=nullptr 和 executor=nullptr 调用接口，验证是否能安全返回错误码而非崩溃。

---

### Bug 5: Inplace 场景下 Cast 操作可能导致数据类型不一致

- **位置**: 第 103-107 行结合第 122 行 `aclnnInplaceLeakyReluGetWorkspaceSize`
- **类型**: 逻辑缺陷
- **严重程度**: 低
- **描述**: Inplace 模式下 `self` 和 `out` 是同一个 tensor。LeakyRelu 的输出经过 `Contiguous` 后可能改变内部存储，再经 `Cast(output, out->GetDataType())` 进行类型转换（实际是同类型转换），虽然功能正确但引入了不必要的计算开销。更关键的是，如果 LeakyRelu kernel 内部改变了输出 dtype（某些平台上 float16 可能提升为 float32 计算），Cast 回原类型时可能引入精度问题。
- **触发条件**: 使用 `aclnnInplaceLeakyRelu` 接口，输入为 DT_FLOAT16 类型。
- **测试方案**: 对比 inplace 与非 inplace 版本在 DT_FLOAT16 输入上的计算结果，验证数值一致性。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 61-63 行 | 逻辑缺陷 | 高 | CheckShape 未校验 out 与 self 的 shape 一致性 |
| 2 | 第 103 行 | 精度损失 | 中 | negativeSlope 强制 ToFloat() 对 DT_DOUBLE 精度丢失 |
| 3 | 第 54-58 行 | 校验不完整 | 中 | 未校验输出 tensor dtype 是否在支持列表 |
| 4 | 第 79-80 行 | 空指针风险 | 中 | 未校验 workspaceSize/executor 指针是否为空 |
| 5 | 第 103-107, 122 行 | 逻辑缺陷 | 低 | Inplace 场景冗余 Cast 及潜在精度问题 |
