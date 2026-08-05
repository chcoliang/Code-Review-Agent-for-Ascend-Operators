# Ascend NPU 算子代码审查报告 - aclnn_leaky_relu.cpp (A104)

## Bug 列表

### Bug 1: 空指针检查失败时错误返回 ACLNN_SUCCESS

- **位置**: 第 69 行
- **类型**: 逻辑错误 / 错误码误用
- **严重程度**: 严重 (Critical)
- **描述**: `CHECK_RET(CheckNotNull(self, negativeSlope, out), ACLNN_SUCCESS);` 中，当 `CheckNotNull` 返回 `false`（即检测到空指针）时，`CHECK_RET` 宏会将第二个参数作为返回值返回。这里第二个参数为 `ACLNN_SUCCESS`，意味着即使输入参数为空指针，函数仍然返回成功状态码。正确做法应返回错误码如 `ACLNN_ERR_PARAM_NULLPTR`。
- **触发条件**: 调用 `aclnnLeakyReluGetWorkspaceSize` 时传入 `self`、`negativeSlope` 或 `out` 为 `nullptr`。此时函数返回成功，调用方误以为参数合法，后续若使用 executor 或 workspaceSize 将导致未定义行为或空指针解引用崩溃。
- **测试方案**:
  ```cpp
  // 传入nullptr，期望返回错误码，实际返回ACLNN_SUCCESS
  aclnnStatus ret = aclnnLeakyReluGetWorkspaceSize(nullptr, negativeSlope, out, &workspaceSize, &executor);
  EXPECT_NE(ret, ACLNN_SUCCESS); // 该断言会失败，暴露bug
  ```

### Bug 2: negativeSlope 精度丢失 (ToFloat 对 double 类型截断)

- **位置**: 第 104 行
- **类型**: 精度损失
- **严重程度**: 中等 (Medium)
- **描述**: `negativeSlope->ToFloat()` 将 scalar 强制转换为 float32。然而该算子支持 `DT_DOUBLE` 数据类型（第 32、35 行），当输入 tensor 为 double 精度且 negativeSlope 也需要 double 精度时，ToFloat() 会将 64 位双精度值截断为 32 位单精度，导致计算精度损失。应根据输入 tensor 的数据类型选择 `ToFloat()` 或 `ToDouble()`。
- **触发条件**: 输入 self 为 DT_DOUBLE 类型，且 negativeSlope 的值超出 float32 精度范围（如极小值 1e-40 或需要高精度的值）。
- **测试方案**:
  ```cpp
  // 构造 double 类型输入，negativeSlope 设为需要 double 精度的值
  aclScalar *slope = aclCreateScalar(1e-40, ACL_DOUBLE);
  // 验证输出结果是否与 CPU double 精度计算结果一致
  // 期望: 结果精确，实际: 精度丢失
  ```

### Bug 3: 输出 tensor 数据类型未校验

- **位置**: 第 54-58 行 / 第 67-78 行
- **类型**: 参数校验不完整
- **严重程度**: 中等 (Medium)
- **描述**: `CheckDtypeValid` 仅校验了输入 `self` 的数据类型是否在支持列表中，但未校验输出 `out` 的数据类型。虽然第 108 行会对输出做 Cast，但如果 `out` 的数据类型是不支持的类型（如 INT8、BOOL 等），Cast 可能在运行时失败或产生无意义结果，且没有明确的错误信息返回给用户。
- **触发条件**: 调用时 out tensor 的 dtype 设置为非浮点类型（如 DT_INT32），校验阶段不会报错，但执行阶段 Cast 可能失败。
- **测试方案**:
  ```cpp
  // out 设为 INT32 类型
  aclTensor *out = CreateTensor(shape, ACL_INT32);
  aclnnStatus ret = aclnnLeakyReluGetWorkspaceSize(self, slope, out, &ws, &exec);
  // 期望: 返回 ACLNN_ERR_PARAM_INVALID（dtype不支持）
  // 实际: 可能返回 SUCCESS 但执行时出错
  ```

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第69行 | 逻辑错误/错误码误用 | 严重 | 空指针检查失败时返回 ACLNN_SUCCESS 而非错误码，导致空指针未被拦截 |
| 2 | 第104行 | 精度损失 | 中等 | negativeSlope 强制 ToFloat() 对 double 场景丢失精度 |
| 3 | 第54-58行 | 参数校验不完整 | 中等 | 仅校验输入 self 的 dtype，未校验输出 out 的 dtype 是否合法 |
