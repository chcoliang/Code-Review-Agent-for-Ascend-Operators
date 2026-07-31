# Code Review: A124 - aclnn_scaled_masked_softmax.cpp

## Bug 1: 支持数据类型列表错误包含 INT32

- **位置**: 第 34-35 行 (`SOFTMAX_X_DTYPE_SUPPORT_LIST` 定义)
- **类型**: 数据类型校验错误
- **严重程度**: 高
- **描述**: `SOFTMAX_X_DTYPE_SUPPORT_LIST` 包含 `op::DataType::DT_INT32`，这是一个整数类型。Softmax 算子的数学定义要求浮点运算（指数函数、除法等），整数类型在语义上不适用且底层 kernel 不支持 INT32 输入。允许 INT32 通过参数校验会导致：
  1. 底层 kernel 收到不支持的类型，可能产生计算错误或设备异常；
  2. 即使不崩溃，INT32 数据经 softmax 运算后结果无数学意义（整数无法精确表示概率分布）。
- **触发条件**: 调用 `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 时，`x` 和 `y` 的 dtype 为 `DT_INT32`。
- **修复建议**: 将第 35 行修改为：
  ```cpp
  static const std::initializer_list<op::DataType> SOFTMAX_X_DTYPE_SUPPORT_LIST = {
      op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
  ```
- **测试方案**:
  1. 构造 INT32 类型的 `x` 和 `y` 张量，调用 GetWorkspaceSize，验证返回 `ACLNN_ERR_PARAM_INVALID`（修复后）。
  2. 在修复前，传入 INT32 张量走完整推理流程，观察是否出现 kernel 执行错误或设备异常。
  3. 验证合法类型（FP32/FP16/BF16）仍能正常通过校验。

---

## 汇总表

| 编号 | 行号 | Bug 类型 | 严重程度 | 简述 |
|------|------|----------|----------|------|
| 1 | 34-35 | 数据类型校验错误 | 高 | 错误包含 `DT_INT32`，允许非浮点类型通过校验，导致 kernel 异常或计算错误 |
