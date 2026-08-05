# Ascend NPU 算子代码审查报告 - aclnn_swish.cpp (A94)

## Bug 列表

### Bug 1: beta 默认值错误 (scale = 0.5f)

- **位置**: 第 109 行
- **类型**: 逻辑错误 / 数学计算错误
- **严重程度**: 高 (High)
- **描述**: Swish 激活函数定义为 `swish(x) = x * sigmoid(beta * x)`，标准 Swish 使用 `beta = 1.0` 作为默认值。但代码中当 `betaOptional` 为 `nullptr` 时，将 `scale` 默认设为 `0.5f` 而非 `1.0f`。这会导致在未指定 beta 参数时，计算结果为 `x * sigmoid(0.5 * x)` 而非标准的 `x * sigmoid(x)`，输出数值全部错误。
- **触发条件**: 用户调用 `aclnnSwish` 时不传入 `betaOptional` 参数（即传入 nullptr），所有未指定 beta 的标准 Swish 调用都会触发。
- **修复建议**: 将 `float scale = 0.5f;` 改为 `float scale = 1.0f;`
- **测试方案**:
  1. 构造输入张量 `x = [1.0, 2.0, -1.0]`，`betaOptional = nullptr`；
  2. 调用 `aclnnSwish`，验证输出是否等于 `x * sigmoid(x)` 即 `[0.7311, 1.7616, -0.2689]`；
  3. 对比当前错误输出 `x * sigmoid(0.5*x)` = `[0.6225, 1.5379, -0.3775]`，确认修复后结果正确。

---

### Bug 2: reshapeLongTensor 中 dimSize 比较逻辑冗余/潜在问题

- **位置**: 第 74 行, 第 118-119 行
- **类型**: 逻辑缺陷
- **严重程度**: 低 (Low)
- **描述**: `reshapeLongTensor` 函数内部判断 `originalDimSize == dimSize && dimSize <= MAX_SUPPORT_DIMS_NUMS` 时返回原张量。但该函数在第 119 行被调用时，外层已判断 `dimSize > MAX_SUPPORT_DIMS_NUMS`，此时传入的 `originalDimSize`（即外层 `dimSize`）必然大于 `MAX_SUPPORT_DIMS_NUMS`。如果 `swishOut` 经过前面的 reshape 后维度已被压缩（`x->GetViewShape().GetDimNum()` != `originalDimSize`），函数会执行 reshape；但如果 swishOut 维度未变（仍等于 originalDimSize），则条件 `dimSize <= MAX_SUPPORT_DIMS_NUMS` 为 false，也会执行 reshape。逻辑上该条件判断存在冗余，不会导致功能错误但降低代码可读性。
- **触发条件**: 输入张量维度超过 `MAX_SUPPORT_DIMS_NUMS` 时。
- **测试方案**: 构造超过 MAX_SUPPORT_DIMS_NUMS 维度的张量（如 9 维），验证 reshape 来回转换后结果正确，shape 与输出一致。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 109 行 | 逻辑错误 | 高 | beta 默认值应为 1.0f，错误设为 0.5f，导致所有默认 Swish 计算结果错误 |
| 2 | 第 74/118-119 行 | 逻辑缺陷 | 低 | reshapeLongTensor 内部条件判断冗余，可读性差 |
