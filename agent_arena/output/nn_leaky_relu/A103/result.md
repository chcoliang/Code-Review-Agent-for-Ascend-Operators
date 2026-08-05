# Ascend NPU 算子代码审查报告 - aclnn_leaky_relu.cpp (A103)

## Bug 列表

### Bug 1: negativeSlope 标量转换精度丢失

- **位置**: 第 104 行 `l0op::LeakyRelu(selfContiguous, negativeSlope->ToFloat(), uniqueExecutor.get())`
- **类型**: 精度丢失 (数据类型处理错误)
- **严重程度**: 中
- **描述**: `negativeSlope->ToFloat()` 将标量无条件转换为 `float`(32位浮点)。当输入张量的数据类型为 `DT_DOUBLE`（64位双精度浮点）时，negativeSlope 的精度从 double 被截断为 float，导致计算结果精度不符合预期。应根据输入 tensor 的 dtype 选择使用 `ToFloat()` 或 `ToDouble()`。
- **触发条件**: 输入张量 dtype 为 `DT_DOUBLE`，且 negativeSlope 的值在 float 精度范围外有有效位（如 `0.123456789012345`）。
- **测试方案**: 构造 DT_DOUBLE 输入张量，设置 negativeSlope 为一个需要双精度才能精确表示的值（如 `1e-15 + 1e-16`），对含负数的输入执行 LeakyReLU，对比输出与 CPU 双精度参考实现的误差是否超出 double 精度容忍范围。

---

### Bug 2: 输出张量 dtype 未校验

- **位置**: 第 54-58 行 `CheckDtypeValid` 函数，及第 67-78 行 `CheckParams` 函数
- **类型**: 参数校验缺失
- **严重程度**: 中
- **描述**: `CheckDtypeValid` 仅检查了输入 `self` 的数据类型是否在支持列表中，但未检查输出 `out` 的数据类型。如果调用者传入一个不支持的输出 dtype（如 `DT_INT8`、`DT_BOOL` 等），第 108 行的 `l0op::Cast(output, out->GetDataType(), ...)` 可能产生未定义行为或内部错误，而非返回明确的参数校验错误码。
- **触发条件**: 调用 `aclnnLeakyReluGetWorkspaceSize` 时，`out` 张量的 dtype 设置为不在算子支持列表中的类型（如 DT_INT8, DT_UINT8, DT_BOOL 等）。
- **测试方案**: 创建输入为 DT_FLOAT 的张量，输出为 DT_INT8 的张量，调用该接口，验证是否返回 `ACLNN_ERR_PARAM_INVALID` 而非内部崩溃或不明确错误。

---

### Bug 3: DT_INT32 类型支持 LeakyReLU 语义正确性存疑

- **位置**: 第 34-35 行 `ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST` 包含 `DT_INT32`，及第 104 行 LeakyRelu 调用
- **类型**: 逻辑错误 / 语义正确性
- **严重程度**: 中
- **描述**: LeakyReLU 定义为 `f(x) = x if x >= 0, negativeSlope * x if x < 0`。当输入为 `DT_INT32` 时，`negativeSlope * x` 的结果为浮点数（如 `0.01 * (-5) = -0.05`），但 LeakyRelu kernel 的输出 dtype 通常与输入一致（INT32），导致结果被截断为 0。这使得对于大部分 negativeSlope < 1 的场景，INT32 负数输入的输出全部变为 0，与 LeakyReLU 语义不符（退化为 ReLU）。
- **触发条件**: 在 Ascend910B 平台上，输入 dtype 为 DT_INT32，negativeSlope 为非整数值（如默认的 0.01），输入包含负数。
- **测试方案**: 在 910B 平台构造 DT_INT32 输入 `[-100, -50, 0, 50, 100]`，negativeSlope=0.01，检查输出是否为 `[-1, 0, 0, 50, 100]`（正确应为 `-1.0, -0.5, 0, 50, 100`），验证整型截断问题。

---

### Bug 4: 输入输出 dtype 不一致时缺乏校验

- **位置**: 第 61-65 行 `CheckShape` 函数及第 67-78 行 `CheckParams` 函数
- **类型**: 参数校验缺失
- **严重程度**: 低
- **描述**: `CheckParams` 未校验输入 `self` 和输出 `out` 的数据类型是否一致或兼容。虽然第 108 行有 Cast 操作来处理类型转换，但 LeakyReLU 的标准语义要求输出 dtype 与输入一致。当 `self` 为 DT_FLOAT16 而 `out` 为 DT_DOUBLE 时，虽然不会崩溃，但隐式类型转换可能不是用户预期行为，且不符合 PyTorch 等框架的 LeakyReLU 语义（输出 dtype 必须等于输入 dtype）。
- **触发条件**: 输入 self 的 dtype 为 DT_FLOAT16，输出 out 的 dtype 为 DT_FLOAT 或 DT_DOUBLE。
- **测试方案**: 构造 DT_FLOAT16 输入和 DT_FLOAT 输出张量，调用接口验证是否应报错或产生非预期的类型提升行为。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 104 行 | 精度丢失 | 中 | negativeSlope 无条件转 float，DT_DOUBLE 场景精度丢失 |
| 2 | 第 54-58 行 | 参数校验缺失 | 中 | 未检查输出张量 out 的 dtype 是否在支持列表内 |
| 3 | 第 34-35 行 | 逻辑/语义错误 | 中 | DT_INT32 支持 LeakyReLU 存在整型截断，语义退化为 ReLU |
| 4 | 第 61-78 行 | 参数校验缺失 | 低 | 未校验输入输出 dtype 一致性，隐式 Cast 不符合标准语义 |
