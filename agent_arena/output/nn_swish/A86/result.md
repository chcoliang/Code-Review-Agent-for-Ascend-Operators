# Ascend NPU 算子代码审查报告 - aclnn_swish.cpp (A86)

## Bug 列表

### Bug 1: 空指针检查返回错误码错误

- **位置**: 第 59 行
- **类型**: 逻辑错误 / 错误码误用
- **严重程度**: 高
- **描述**: `CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_SUCCESS)` 中第二个参数应为检查失败时返回的错误码。当 `self` 或 `out` 为空指针时，`CheckNotNull2Tensor` 返回 false，此时 `CHECK_RET` 宏会返回第二个参数 `ACLNN_SUCCESS`，意味着空指针场景下函数返回成功，完全丧失了空指针保护能力。正确写法应为 `CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR)` 或 `ACLNN_ERR_PARAM_INVALID`。
- **触发条件**: 调用 `aclnnSwishGetWorkspaceSize` 时传入 `self=nullptr` 或 `out=nullptr`。
- **测试方案**: 构造测试用例，将 self 或 out 设为 nullptr 调用 `aclnnSwishGetWorkspaceSize`，验证返回值应为错误码而非 ACLNN_SUCCESS；继续执行后续逻辑将导致空指针解引用崩溃。

---

### Bug 2: ReshapeSelfValueGetActivation 传入参数不一致

- **位置**: 第 107 行
- **类型**: 接口调用参数错误
- **严重程度**: 中
- **描述**: `ReshapeSelfValueGetActivation(self, dimSize, selfContiguous, uniqueExecutor)` 第一个参数传入了原始的 `self` 而非连续化后的 `selfContiguous`。根据上下文逻辑，`dimSize` 在第 105 行从 `self->GetViewShape().GetDimNum()` 获取，而 reshape 操作应基于连续化后的张量进行。如果 `self` 非连续且 shape view 不同于 selfContiguous，可能导致 reshape 维度信息不匹配。此外，传入 `uniqueExecutor`（智能指针对象）而非 `uniqueExecutor.get()`（原始指针），与其他调用点（如第 102、114、119、122 行均使用 `.get()`）风格不一致，可能存在类型不匹配的编译隐患或隐式转换问题。
- **触发条件**: 输入 tensor `self` 维度数大于 `MAX_SUPPORT_DIMS_NUMS` 且非连续存储时触发。
- **测试方案**: 构造一个维度数超过 8（MAX_SUPPORT_DIMS_NUMS）且非连续的输入 tensor，验证 reshape 后的维度信息是否正确；检查编译期是否有隐式转换警告。

---

### Bug 3: reshapeLongTensor 中 valuePerm 为 nullptr 时传递给 Reshape

- **位置**: 第 71-79 行
- **类型**: 潜在空指针传递
- **严重程度**: 低
- **描述**: 函数 `reshapeLongTensor` 的参数 `valuePerm` 默认值为 `nullptr`。当 `dimSize > MAX_SUPPORT_DIMS_NUMS` 但直接调用该函数未传入 `valuePerm` 时（虽然当前代码第 119 行传入了 `shapeOriDetial`），`l0op::Reshape(x, valuePerm, executor)` 会收到空指针作为目标 shape，可能导致未定义行为。函数接口设计存在防御性不足的问题。
- **触发条件**: 若未来代码修改中调用 `reshapeLongTensor` 时未传入 valuePerm 参数。
- **测试方案**: 代码审查确认所有调用路径均传入有效的 valuePerm；添加 nullptr 检查防御。

---

### Bug 4: shapeOriDetial 未做空指针检查即使用

- **位置**: 第 106 行获取，第 119 行使用
- **类型**: 缺少空指针检查
- **严重程度**: 中
- **描述**: `GetTensorShapeActivation(selfContiguous, uniqueExecutor.get())` 返回的 `shapeOriDetial` 未进行空指针检查，直接在第 119 行传入 `reshapeLongTensor`。如果 `GetTensorShapeActivation` 内部分配失败返回 nullptr，后续 `l0op::Reshape` 将收到无效参数。
- **触发条件**: 内存分配失败或 selfContiguous 的 shape 信息异常时。
- **测试方案**: 模拟内存不足场景，验证 `GetTensorShapeActivation` 返回 nullptr 时是否有合理的错误处理；添加 `CHECK_RET(shapeOriDetial != nullptr, ACLNN_ERR_INNER_NULLPTR)` 检查。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第59行 | 逻辑错误/错误码误用 | 高 | 空指针检查失败时返回 ACLNN_SUCCESS，丧失保护 |
| 2 | 第107行 | 接口调用参数错误 | 中 | 传入 self 而非 selfContiguous，executor 传参方式不一致 |
| 3 | 第71-79行 | 潜在空指针传递 | 低 | valuePerm 默认 nullptr 传给 Reshape 缺乏防御 |
| 4 | 第106/119行 | 缺少空指针检查 | 中 | shapeOriDetial 未校验直接使用 |
