# Ascend NPU 算子代码审查报告 - aclnn_leaky_relu.cpp (A107)

## Bug 列表

### Bug 1: 缺少 Cast 类型转换操作

- **位置**: 第 108 行
- **类型**: 逻辑缺陷 / 功能缺失
- **严重程度**: 高
- **描述**: 代码注释明确说明"将计算结果转换成输出out的数据类型"，但实际代码只是简单赋值 `auto castOut = output;`，并未调用 `l0op::Cast()` 进行类型转换。当输入 `self` 的数据类型与输出 `out` 的数据类型不一致时（例如输入为 float16，输出为 float），计算结果不会进行类型转换，导致输出数据类型错误或数据损坏。
- **触发条件**: 当调用者指定的输出 tensor `out` 的 dtype 与输入 tensor `self` 的 dtype 不同时触发。例如输入为 DT_FLOAT16，输出为 DT_FLOAT。
- **测试方案**: 构造输入 tensor 为 float16 类型，输出 tensor 为 float 类型，调用 aclnnLeakyRelu，验证输出结果的数据类型和数值是否正确。
- **修复建议**:
```cpp
// 应改为:
auto castOut = l0op::Cast(output, out->GetDataType(), uniqueExecutor.get());
```

### Bug 2: negativeSlope 精度丢失

- **位置**: 第 104 行
- **类型**: 精度问题
- **严重程度**: 中
- **描述**: `negativeSlope->ToFloat()` 将标量强制转换为 float (32位浮点)。当输入 tensor 为 DT_DOUBLE 类型时，LeakyRelu 的 negative slope 参数精度被截断为 32 位，导致高精度场景下计算结果不准确。算子支持列表中明确包含 DT_DOUBLE 类型，但 slope 参数未能保持对应精度。
- **触发条件**: 输入 tensor 为 DT_DOUBLE 类型，且 negativeSlope 的值需要超过 float 精度才能准确表示（例如非常小的值如 1e-40 或有大量有效位数的值）。
- **测试方案**: 构造 DT_DOUBLE 类型输入，设置 negativeSlope 为需要 double 精度表示的值（如 0.123456789012345678），对负数输入验证输出精度是否满足 double 精度要求。

### Bug 3: 输出 tensor 数据类型未校验

- **位置**: 第 67-78 行 (`CheckParams` 函数)
- **类型**: 参数校验不完整
- **严重程度**: 中
- **描述**: `CheckDtypeValid` 仅检查了输入 `self` 的数据类型是否在支持列表中，但未检查输出 `out` 的数据类型是否合法。如果用户传入不支持的输出类型（如 DT_INT32），由于缺少 Cast 操作（Bug 1），可能导致未定义行为；即使修复了 Cast，不支持的目标类型也可能导致 Cast 失败。
- **触发条件**: 用户传入数据类型为非支持类型（如 DT_INT8、DT_INT32）的输出 tensor。
- **测试方案**: 构造输出 tensor 为 DT_INT32 类型，验证是否返回合适的错误码而非崩溃。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 108 行 | 逻辑缺陷 | 高 | 缺少 Cast 类型转换，仅赋值未做 dtype 转换 |
| 2 | 第 104 行 | 精度问题 | 中 | negativeSlope 强转 float 导致 double 场景精度丢失 |
| 3 | 第 67-78 行 | 校验缺失 | 中 | 输出 tensor 的数据类型未做合法性校验 |
