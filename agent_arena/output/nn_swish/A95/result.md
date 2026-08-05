# Ascend NPU 算子代码审查报告 - aclnn_swish.cpp (A95)

## Bug 列表

### Bug 1: betaOptional 参数未生效，Swish 调用使用硬编码常量

- **位置**: 第 109-114 行
- **类型**: 逻辑错误 (参数未使用)
- **严重程度**: 高 (Critical)
- **描述**: 代码在第 109-112 行从 `betaOptional` 中正确提取了 `scale` 值，但在第 114 行调用 `l0op::Swish` 时，第二个参数硬编码为 `1.0f`，而不是使用计算得到的 `scale` 变量。这导致无论用户传入什么 `beta` 值，Swish 算子始终按 `beta=1.0` 执行，即 `x * sigmoid(1.0 * x)` 而非正确的 `x * sigmoid(beta * x)`。
- **触发条件**: 当用户传入非空且值不为 1.0 的 `betaOptional` 参数时，计算结果将与预期不符。例如传入 `beta=0.5`，期望 `x * sigmoid(0.5x)` 但实际得到 `x * sigmoid(x)`。
- **修复建议**: 将第 114 行 `l0op::Swish(reshapeSelf, 1.0f, uniqueExecutor.get())` 改为 `l0op::Swish(reshapeSelf, scale, uniqueExecutor.get())`。
- **测试方案**:
  1. 构造输入 tensor `self = [1.0, 2.0, 3.0]`，设置 `betaOptional = 2.0`
  2. 调用 `aclnnSwish`，验证输出是否等于 `x * sigmoid(2.0 * x)`
  3. 对比 `beta=1.0` 和 `beta=2.0` 的输出，确认结果不同
  4. 边界测试：`beta=0`，期望输出为 `0.5 * x`

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 114 行 | 逻辑错误 | 高 | `scale` 变量未传入 `l0op::Swish`，硬编码 `1.0f` 导致 `betaOptional` 参数完全失效 |
