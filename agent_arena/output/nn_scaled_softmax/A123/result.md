# Code Review: A123 - aclnn_scaled_masked_softmax.cpp

## Bug 1: 支持数据类型列表缺少 BF16

- **位置**: 第 34-35 行 (`SOFTMAX_X_DTYPE_SUPPORT_LIST` 定义)
- **类型**: 数据类型支持不完整
- **严重程度**: 高
- **描述**: `SOFTMAX_X_DTYPE_SUPPORT_LIST` 仅包含 `{DT_FLOAT, DT_FLOAT16}`，缺少 `DT_BF16`。Ascend 910B 平台的 ScaledMaskedSoftmax 算子应支持 BF16 数据类型（在其他正确版本中均包含 `op::DataType::DT_BF16`）。这导致用户传入 BF16 格式的输入张量时，参数校验会错误地拒绝合法输入，返回 `ACLNN_ERR_PARAM_INVALID`。
- **触发条件**: 调用 `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 时，`x` 或 `y` 的 dtype 为 `DT_BF16`。
- **修复建议**: 将第 35 行修改为：
  ```cpp
  static const std::initializer_list<op::DataType> SOFTMAX_X_DTYPE_SUPPORT_LIST = {
      op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
  ```
- **测试方案**:
  1. 构造 BF16 类型的 `x` 和 `y` 张量，调用 GetWorkspaceSize，验证返回 `ACLNN_SUCCESS`。
  2. 对比 FP32/FP16/BF16 三种类型的端到端推理结果，确认 BF16 路径精度正确。

---

## 汇总表

| 编号 | 行号 | Bug 类型 | 严重程度 | 简述 |
|------|------|----------|----------|------|
| 1 | 34-35 | 数据类型支持不完整 | 高 | 缺少 `DT_BF16` 支持，导致 BF16 输入被错误拒绝 |
