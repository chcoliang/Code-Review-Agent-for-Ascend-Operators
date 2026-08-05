# Ascend NPU 算子代码审查报告 - aclnn_swish.cpp (A92)

## Bug 列表

### Bug 1: ReshapeSelfValueGetActivation 使用了非连续的原始 tensor

- **位置**: 第 106 行
- **类型**: 逻辑错误 / 参数传递错误
- **严重程度**: 高
- **描述**: 第 101 行已经通过 `l0op::Contiguous(self, ...)` 将 `self` 转换为连续内存的 `selfContiguous`，但第 106 行调用 `ReshapeSelfValueGetActivation` 时第一个参数仍传入了原始的 `self` 而非 `selfContiguous`。如果输入 tensor 的内存布局是非连续的（如经过 slice/transpose 操作），后续基于 `self` 进行的 reshape 计算可能得到错误的数据视图或触发非法内存访问。
- **触发条件**: 输入 tensor `self` 为非连续存储（如经过 transpose、narrow、slice 等操作后未显式调用 contiguous 的 tensor），且维度数超过 `MAX_SUPPORT_DIMS_NUMS`。
- **修复建议**: 将第 106 行改为 `ReshapeSelfValueGetActivation(selfContiguous, dimSize, selfContiguous, uniqueExecutor)`。
- **测试方案**: 构造一个经过 transpose 后非连续的高维 tensor（维度 > 8），调用 aclnnSwish，验证输出结果的正确性与是否发生内存越界。

---

### Bug 2: CheckDim 未检查输入 tensor self 的维度

- **位置**: 第 52-55 行
- **类型**: 校验遗漏
- **严重程度**: 中
- **描述**: `CheckDim` 函数接收了 `self` 和 `out` 两个参数，但函数体内只对 `out` 做了 `OP_CHECK_MAX_DIM` 检查，完全忽略了对 `self` 的维度校验。虽然后续代码对超过 `MAX_SUPPORT_DIMS_NUMS` 的 `self` 有 reshape 逻辑处理，但 `out` 的维度限制与 `self` 不一致会导致：当 `self` 维度 > MAX 但 `out` 维度 <= MAX 时通过校验，但后续 ViewCopy 时 shape 不匹配可能导致错误。
- **触发条件**: 输入 `self` 维度超出支持上限，但 `out` 维度在范围内（shape 不一致场景）。
- **修复建议**: 在 `CheckDim` 中增加对 `self` 的维度检查，或明确文档化此处只限制 `out` 的设计意图。
- **测试方案**: 传入维度数超过 MAX_SUPPORT_DIMS_NUMS 的 self tensor 和正常维度的 out tensor，观察是否正确报错或正确执行。

---

### Bug 3: reshapeLongTensor 中 valuePerm 默认为 nullptr 存在空指针风险

- **位置**: 第 70-79 行
- **类型**: 空指针风险
- **严重程度**: 中
- **描述**: `reshapeLongTensor` 函数的参数 `valuePerm` 默认值为 `nullptr`。当 `originalDimSize != dimSize` 或 `dimSize > MAX_SUPPORT_DIMS_NUMS` 时，函数会跳过 early return 并调用 `l0op::Reshape(x, valuePerm, executor)`。如果调用时未传入有效的 `valuePerm`（使用默认的 nullptr），则会将 nullptr 传给 Reshape，可能导致空指针解引用或未定义行为。
- **触发条件**: 调用 `reshapeLongTensor` 时未传入第四个参数 `valuePerm`，且 tensor 的实际维度与 `originalDimSize` 不同。
- **修复建议**: 在调用 `l0op::Reshape` 之前增加 `valuePerm` 的空指针检查，或移除默认参数强制调用者提供有效值。
- **测试方案**: 在单元测试中模拟不传递 valuePerm 参数的调用路径，验证是否发生崩溃。

---

### Bug 4: dimSize 取自原始 self 而非 selfContiguous

- **位置**: 第 104 行
- **类型**: 潜在逻辑缺陷
- **严重程度**: 低
- **描述**: `dimSize` 从 `self->GetViewShape().GetDimNum()` 获取，而非从 `selfContiguous` 获取。通常 Contiguous 操作不改变维度数，但在某些边缘情况下（如 tensor 元数据与实际存储不一致），使用原始 tensor 的维度信息可能存在不一致风险。更重要的是，后续第 105 行 `GetTensorShapeActivation` 使用了 `selfContiguous`，而第 106 行的 reshape 逻辑却基于 `self` 的 dimSize，语义不统一。
- **触发条件**: 极端边缘情况下 self 的 view shape 与 contiguous 后 shape 不完全一致时。
- **修复建议**: 统一从 `selfContiguous` 获取 `dimSize`。
- **测试方案**: 构造 view shape 与 storage shape 不一致的 tensor 输入，检查维度获取是否正确。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| Bug 1 | 第 106 行 | 逻辑错误/参数错误 | 高 | Reshape 使用非连续原始 tensor 而非 selfContiguous |
| Bug 2 | 第 52-55 行 | 校验遗漏 | 中 | CheckDim 未检查 self 维度，仅检查 out |
| Bug 3 | 第 70-79 行 | 空指针风险 | 中 | reshapeLongTensor 的 valuePerm 默认 nullptr 可能传入 Reshape |
| Bug 4 | 第 104 行 | 潜在逻辑缺陷 | 低 | dimSize 取自原始 self 而非 contiguous 后的 tensor |
