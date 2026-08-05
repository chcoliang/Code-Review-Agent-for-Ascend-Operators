# Code Review: aclnn_scaled_masked_softmax.cpp (A137)

## Bug 列表

### Bug 1: CheckNotNull 缺少对输出张量 y 的空指针检查

- **位置**: 第 46-52 行, `CheckNotNull` 函数
- **类型**: 空指针解引用风险
- **严重程度**: 高
- **描述**: 函数签名接收 `aclTensor* y` 参数，但函数体内只检查了 `x` 和 `mask` 是否为空，未对 `y` 进行空指针检查。后续 `CheckDtypeValid` 和 `CheckTensorDim` 中都会对 `y` 解引用（如 `y->GetViewShape()`），若 `y` 为 nullptr 将导致段错误崩溃。
- **触发条件**: 调用 `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 时传入 `y = nullptr`。
- **测试方案**: 构造单元测试，传入有效的 `x`、`mask` 和 `scale`，但将 `y` 设为 `nullptr`，验证函数是否返回 `ACLNN_ERR_INNER_NULLPTR` 而非崩溃。

---

### Bug 2: 错误日志中 D_LIMIT 范围描述与实际逻辑不一致

- **位置**: 第 104-106 行, `CheckShape` 函数
- **类型**: 日志/文档错误
- **严重程度**: 低
- **描述**: 当平台为 ASCEND910_95 时，`dDimLimit` 被设置为 8192，但错误日志始终打印 `"Expected x and mask dim4 in range of (0, 4096]"`，未反映实际的 8192 上限。这会误导用户排查问题。
- **触发条件**: 在 ASCEND910_95 平台上，当 `x` 的第4维大小在 (4096, 8192] 范围内时，虽然合法但若超过 8192 报错信息仍显示 4096 上限。
- **测试方案**: 在 ASCEND910_95 平台上传入 dim3 > 8192 的张量，检查错误日志是否正确显示 8192 限制。

---

### Bug 3: `extern "C"` 块内包含了使用 C++ 特性的匿名命名空间和模板代码

- **位置**: 第 25-27 行 `extern "C" {` 与第 29-123 行匿名 namespace
- **类型**: 语言链接规范错误
- **严重程度**: 中
- **描述**: `extern "C"` 块从第 26 行开始，包裹了内部使用 `std::initializer_list`、`namespace`、模板等 C++ 特性的代码。虽然大多数编译器对 `extern "C"` 内的 C++ 代码仍能正确编译（仅影响链接符号修饰），但匿名命名空间和 C++ 类型放在 `extern "C"` 块中违反语义规范，可能导致某些编译器警告或未定义行为。正确做法是仅将对外暴露的 C 接口函数声明/定义放在 `extern "C"` 中。
- **触发条件**: 使用严格 C++ 标准检查的编译器编译该文件。
- **测试方案**: 使用 `-Wall -Wpedantic` 编译选项，检查是否有 linkage 相关警告。

---

### Bug 4: fixedTriuMask 参数硬编码为 false 传递，忽略用户意图

- **位置**: 第 131-135 行, `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 函数
- **类型**: 逻辑缺陷/功能限制
- **严重程度**: 中
- **描述**: 当 `fixedTriuMask` 为 `true` 时直接报错返回，当为 `false` 时传递字面量 `false` 给内部函数。这意味着该参数完全无用，但接口定义暴露了它。如果将来需要支持 `fixedTriuMask=true`，当前实现的错误信息中 "suppport" 拼写为错误（应为 "support"），且仅靠日志提示不够友好。虽然这可能是有意的功能限制，但拼写错误是确定的 bug。
- **触发条件**: 用户传入 `fixedTriuMask = true`。
- **测试方案**: 传入 `fixedTriuMask = true`，验证返回错误码正确；检查错误日志拼写。

---

### Bug 5: CheckShape 中未校验 x 与 y 的 shape 一致性

- **位置**: 第 74-110 行, `CheckShape` 函数；第 112-122 行, `CheckParams` 函数
- **类型**: 输入校验遗漏
- **严重程度**: 高
- **描述**: `CheckShape` 只验证了 `x` 与 `mask` 之间的形状关系，但从未检查输出张量 `y` 的 shape 是否与 `x` 一致。对于 scaled_masked_softmax 操作，输出 `y` 必须与输入 `x` 形状相同。如果 `y` 的形状与 `x` 不匹配，可能导致内存越界写入或计算结果错误。
- **触发条件**: 传入形状不匹配的 `x`（如 [2,4,8,64]）和 `y`（如 [1,1,1,1]）。
- **测试方案**: 构造 `x` 和 `y` shape 不同的测试用例，验证是否能正确检测并报错。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | L46-52 | 空指针解引用 | 高 | `CheckNotNull` 未检查输出张量 `y` 的空指针 |
| 2 | L104-106 | 日志错误 | 低 | 错误消息硬编码 4096，未适配 8192 平台限制 |
| 3 | L25-27 | 链接规范 | 中 | `extern "C"` 错误包裹 C++ 命名空间和模板代码 |
| 4 | L132 | 拼写/逻辑 | 中 | 错误消息 "suppport" 拼写错误，参数功能受限 |
| 5 | L74-122 | 校验遗漏 | 高 | 缺少输出张量 `y` 与输入 `x` 的 shape 一致性校验 |
