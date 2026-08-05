# Ascend NPU 算子代码审查报告 - aclnn_swish.cpp (A89)

## Bug 列表

### Bug 1: 输出张量数据类型未校验

- **位置**: 第 62 行
- **类型**: 参数校验缺陷
- **严重程度**: 高
- **描述**: `CheckDtypeValidActivation(self, self, supportList)` 将 `self` 传递了两次，第二个参数应为 `out`。这导致输出张量 `out` 的数据类型从未被校验，如果 `out` 的 dtype 不在支持列表中，算子会在计算阶段产生未定义行为或静默错误结果。
- **触发条件**: 传入一个 dtype 不在支持列表（如 DT_INT32、DT_DOUBLE）的输出张量 `out`，而 `self` 的 dtype 合法。
- **测试方案**: 构造 `self` 为 FP16 张量，`out` 为 INT32 张量，调用 `aclnnSwishGetWorkspaceSize`，预期返回 `ACLNN_ERR_PARAM_INVALID`，实际会返回 `ACLNN_SUCCESS` 并继续执行。

---

### Bug 2: ReshapeSelfValueGetActivation 传入原始 self 而非 selfContiguous

- **位置**: 第 107 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `ReshapeSelfValueGetActivation(self, dimSize, selfContiguous, uniqueExecutor)` 的第一个参数传入了原始的 `self`（可能非连续），而不是已经做过 Contiguous 处理的 `selfContiguous`。对于非连续输入张量，在后续 Reshape 操作中会基于错误的 stride/offset 信息进行操作，导致数据错乱或计算错误。
- **触发条件**: 输入 `self` 是一个非连续张量（如通过 slice、transpose 等操作得到的 view 张量），且维度数超过 `MAX_SUPPORT_DIMS_NUMS`。
- **测试方案**: 对一个通过 `transpose` 得到的非连续高维张量调用 swish 算子，对比输出结果与 PyTorch 参考实现，验证结果是否一致。

---

### Bug 3: reshapeLongTensor 中 valuePerm 默认为 nullptr 可能导致非法访问

- **位置**: 第 71-79 行
- **类型**: 空指针风险
- **严重程度**: 中
- **描述**: `reshapeLongTensor` 函数的 `valuePerm` 参数默认值为 `nullptr`。当 `originalDimSize != dimSize`（即张量维度不一致）时，会跳过 early return 进入 `l0op::Reshape(x, valuePerm, executor)` 调用，此时若 `valuePerm` 为空，Reshape 操作将无法获得正确的目标形状信息，可能导致空指针解引用或形状推断错误。
- **触发条件**: 在第 119 行调用时传入了有效的 `shapeOriDetial`，该路径通常安全；但若该函数被其他地方以默认参数调用，或 `shapeOriDetial` 为 nullptr，则触发问题。
- **测试方案**: 构造 `dimSize > MAX_SUPPORT_DIMS_NUMS` 且 `GetTensorShapeActivation` 返回 nullptr 的场景，验证是否崩溃。

---

### Bug 4: reshapeLongTensor 条件逻辑与调用场景不匹配

- **位置**: 第 74 行及第 118-120 行
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: `reshapeLongTensor` 中 early return 条件为 `originalDimSize == dimSize && dimSize <= MAX_SUPPORT_DIMS_NUMS`，意味着当 `originalDimSize != dimSize` 时也会执行 Reshape。但在第 118 行，恢复形状的调用仅在 `dimSize > MAX_SUPPORT_DIMS_NUMS` 时才执行。如果 `ReshapeSelfValueGetActivation` 在 `dimSize <= MAX_SUPPORT_DIMS_NUMS` 时也改变了维度数，则输出不会被恢复为原始形状，导致写入 `out` 时形状不匹配。
- **触发条件**: `self` 的 `dimSize <= MAX_SUPPORT_DIMS_NUMS`，但 `ReshapeSelfValueGetActivation` 内部改变了张量维度（如合并维度优化）。
- **测试方案**: 使用维度数恰好等于 `MAX_SUPPORT_DIMS_NUMS` 的张量，且有可被合并优化的连续维度，验证输出形状是否正确。

---

### Bug 5: uniqueExecutor 传递方式不一致

- **位置**: 第 107 行
- **类型**: 接口调用不一致 / 潜在编译或运行时错误
- **严重程度**: 低
- **描述**: 其他所有 l0op 调用（如第 102、114、122 行）均使用 `uniqueExecutor.get()` 获取原始指针，但第 107 行 `ReshapeSelfValueGetActivation` 直接传入 `uniqueExecutor`（智能指针/包装器对象）。如果该函数期望原始指针类型，则会产生类型不匹配的编译错误；如果有隐式转换重载，则可能导致语义差异（如所有权转移）。
- **触发条件**: 编译时即可发现（若类型不匹配），或在运行时因错误的 executor 状态导致异常。
- **测试方案**: 检查 `ReshapeSelfValueGetActivation` 的函数签名，确认参数类型是否与 `uniqueExecutor` 的类型兼容；若不兼容则应改为 `uniqueExecutor.get()`。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第62行 | 参数校验缺陷 | 高 | `CheckDtypeValidActivation` 传入 `self, self`，未校验 `out` 的 dtype |
| 2 | 第107行 | 逻辑错误 | 高 | Reshape 使用原始 `self` 而非 `selfContiguous`，非连续张量会出错 |
| 3 | 第71-79行 | 空指针风险 | 中 | `valuePerm` 默认 nullptr，特定路径下 Reshape 可能解引用空指针 |
| 4 | 第74行/118行 | 逻辑错误 | 中 | Reshape 恢复条件与前置 Reshape 触发条件不对称，可能丢失形状恢复 |
| 5 | 第107行 | 接口调用不一致 | 低 | `uniqueExecutor` 未调用 `.get()`，与其他调用风格不一致 |
