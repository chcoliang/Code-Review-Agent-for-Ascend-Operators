# Ascend NPU 算子代码审查报告 - aclnn_swish.cpp

## Bug 列表

### Bug 1: 支持数据类型列表包含不合理的 DT_INT32

- **位置**: 第 40 行
- **类型**: 逻辑错误 / 数据类型支持错误
- **严重程度**: 高
- **描述**: `ASCEND910B_DTYPE_SUPPORT_LIST` 中包含 `op::DataType::DT_INT32`。Swish 函数定义为 `x * sigmoid(beta * x)`，其中 sigmoid 输出为浮点数 (0,1) 区间，对整型输入进行 Swish 计算在数学上不合理：sigmoid 中间结果需要浮点表示，整数乘以 (0,1) 区间的浮点数后截断为整数会导致严重精度损失，大部分结果趋近于 0 或原值。这不符合 Swish 的数学语义。
- **触发条件**: 用户传入 DT_INT32 类型的输入张量在 Ascend 910B 平台上调用 aclnnSwish。
- **测试方案**: 构造 DT_INT32 类型输入张量（如 [1, 2, 3, -1, -2]），调用 aclnnSwish，验证输出是否符合 Swish 数学定义，对比浮点计算结果确认精度偏差是否超出可接受范围。

### Bug 2: ReshapeSelfValueGetActivation 传入参数类型可能不匹配

- **位置**: 第 107 行
- **类型**: 接口调用错误
- **严重程度**: 高
- **描述**: `ReshapeSelfValueGetActivation(self, dimSize, selfContiguous, uniqueExecutor)` 的最后一个参数传入的是 `uniqueExecutor`（智能指针/RAII 对象本身），而本文件中其他所有 l0op 调用（如第 102、114、119、122 行）均传入 `uniqueExecutor.get()` 获取裸指针。如果该函数期望的是 `aclOpExecutor*` 类型，则此处传入智能指针对象会导致编译错误或隐式转换的未定义行为。即使函数有重载接受智能指针引用，传入非 const 引用也可能导致所有权转移等副作用。
- **触发条件**: 编译时若类型不匹配会报错；若存在隐式转换则运行时可能导致 executor 状态异常。
- **测试方案**: 检查 `ReshapeSelfValueGetActivation` 函数签名，确认第四个参数类型。将 `uniqueExecutor` 改为 `uniqueExecutor.get()` 后进行编译和功能测试。

### Bug 3: reshapeLongTensor 中 valuePerm 为 nullptr 时可能导致空指针传递

- **位置**: 第 71-79 行（函数定义），第 119 行（调用处）
- **类型**: 潜在空指针风险
- **严重程度**: 中
- **描述**: `reshapeLongTensor` 函数的 `valuePerm` 参数默认值为 `nullptr`。在第 119 行调用时传入了 `shapeOriDetial`，该值来自第 106 行 `GetTensorShapeActivation` 的返回值。如果 `GetTensorShapeActivation` 在某些边界条件下（如 selfContiguous 形状异常）返回 nullptr，则 `l0op::Reshape(x, valuePerm, executor)` 将接收到空指针作为目标形状，可能导致崩溃。函数内部缺少对 `valuePerm` 的空指针检查。
- **触发条件**: 输入张量维度 > MAX_SUPPORT_DIMS_NUMS 且 `GetTensorShapeActivation` 返回 nullptr 时。
- **测试方案**: 构造维度数超过 MAX_SUPPORT_DIMS_NUMS 的输入张量，验证 `shapeOriDetial` 是否为空；在 `reshapeLongTensor` 中添加 nullptr 检查。

### Bug 4: reshapeLongTensor 逻辑条件存在冗余/错误

- **位置**: 第 74 行
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: 条件 `if (originalDimSize == dimSize && dimSize <= MAX_SUPPORT_DIMS_NUMS)` 的语义是"当输出维度等于原始维度且不超过最大支持维度时直接返回"。但该函数仅在第 119 行 `dimSize > MAX_SUPPORT_DIMS_NUMS` 条件下被调用，此时 `originalDimSize`（即原始的 dimSize）大于 MAX_SUPPORT_DIMS_NUMS。传入的 `swishOut` 经过 Swish 计算后的维度 (`x->GetViewShape().GetDimNum()`) 已被 reshape 为 <= MAX_SUPPORT_DIMS_NUMS，因此 `originalDimSize == dimSize` 不会为 true（originalDimSize > MAX 而 dimSize <= MAX）。这意味着该 early return 永远不会触发，属于死代码，但不影响正确性（只是无法提前返回优化）。真正的问题是：如果 swishOut 的维度恰好等于 originalDimSize（未被正确 reshape），该函数会错误地跳过二次 reshape。
- **触发条件**: swishOut 计算后维度未被改变、仍等于原始高维度数时。
- **测试方案**: 构造高维输入，打印 swishOut 的实际维度，验证 reshape 是否被正确执行。

### Bug 5: dimSize 取自原始 self 而非 selfContiguous

- **位置**: 第 105 行
- **类型**: 逻辑隐患
- **严重程度**: 低
- **描述**: `size_t dimSize = self->GetViewShape().GetDimNum()` 使用原始输入 `self` 获取维度数，而后续操作基于 `selfContiguous`。通常 Contiguous 操作不会改变维度数，但在某些 format 转换场景下（如 NCHW 到 NC1HWC0 等私有格式），维度数可能发生变化。使用 `self` 的维度数可能与 `selfContiguous` 实际维度不一致。
- **触发条件**: 输入张量为非连续的私有格式且 Contiguous 后维度数发生变化时。
- **测试方案**: 使用非连续的特殊格式张量（如 FRACTAL_NZ）作为输入，验证 dimSize 与 selfContiguous 的实际维度是否一致。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 40 行 | 数据类型支持错误 | 高 | DT_INT32 不适用于 Swish 运算 |
| 2 | 第 107 行 | 接口调用错误 | 高 | uniqueExecutor 应传 .get() 获取裸指针 |
| 3 | 第 71-79, 119 行 | 空指针风险 | 中 | reshapeLongTensor 缺少 valuePerm 空指针检查 |
| 4 | 第 74 行 | 逻辑错误 | 中 | early return 条件在实际调用场景下永远不触发 |
| 5 | 第 105 行 | 逻辑隐患 | 低 | dimSize 应取自 selfContiguous 而非 self |
