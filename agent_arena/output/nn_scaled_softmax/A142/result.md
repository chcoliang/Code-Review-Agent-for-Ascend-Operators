# Code Review: aclnn_scaled_masked_softmax.cpp (A142)

## Bug 1: mask 数据类型未校验

- **位置**: 第 55-63 行 (`CheckDtypeValid` 函数)
- **类型**: 逻辑遗漏
- **严重程度**: 高
- **描述**: `CheckDtypeValid` 函数定义了 `MASK_DTYPE_SUPPORT_LIST`（第 37 行，仅支持 `DT_BOOL`），但在函数体内从未对 `mask` 参数的数据类型进行校验。参数 `mask` 虽然被传入但未被使用，导致非 BOOL 类型的 mask 张量可以绕过校验进入后续计算，可能导致计算结果错误或内存越界。
- **触发条件**: 传入非 `DT_BOOL` 类型的 mask 张量（如 `DT_FLOAT`、`DT_INT32` 等）。
- **测试方案**: 构造一个 dtype 为 `DT_FLOAT` 或 `DT_INT32` 的 mask 张量传入 `aclnnScaledMaskedSoftmaxGetWorkspaceSize`，验证是否返回 `ACLNN_ERR_PARAM_INVALID`。

## Bug 2: 错误日志中 D_LIMIT 值硬编码，与实际限制不一致

- **位置**: 第 104-106 行
- **类型**: 日志信息错误
- **严重程度**: 低
- **描述**: 当平台为 `ASCEND910_95` 时，`dDimLimit` 实际为 8192，但错误日志始终打印 `"Expected x and mask dim4 in range of (0, 4096]"`，未反映真实的上限值。这会导致用户在 910_95 平台上收到误导性的错误信息。
- **触发条件**: 在 ASCEND910_95 平台上，x 的第 4 维大小超过 8192 时触发错误日志，但日志中显示的限制为 4096。
- **测试方案**: 在 ASCEND910_95 平台上传入 dim3 大小为 5000 的张量（合法值），验证不报错；传入 dim3 为 9000 的张量，验证报错信息中应显示 8192 而非 4096。

## Bug 3: x 与 y 的 shape 一致性未校验

- **位置**: 第 74-110 行 (`CheckShape` 函数) 及第 112-121 行 (`CheckParams` 函数)
- **类型**: 逻辑遗漏
- **严重程度**: 中
- **描述**: `CheckShape` 仅校验了 `x` 和 `mask` 之间的 shape 关系，但从未校验输出张量 `y` 的 shape 是否与 `x` 一致。若 `y` 的 shape 与 `x` 不匹配，写入时可能导致内存越界或计算结果被截断。
- **触发条件**: 传入的 `y` 张量 shape 与 `x` 不同（如 `x` 为 [2,4,8,64]，`y` 为 [1,1,1,1]）。
- **测试方案**: 构造 shape 不一致的 `x` 和 `y` 张量，调用接口验证是否返回错误或产生内存越界。

## Bug 4: fixedTriuMask 参数被硬编码忽略，接口语义不一致

- **位置**: 第 131-135 行
- **类型**: 接口设计缺陷
- **严重程度**: 中
- **描述**: `fixedTriuMask` 参数在接口中暴露给用户，但当其为 `true` 时仅返回错误。且在调用内部函数时始终传入 `false`（第 135 行）。如果该参数永远不支持 `true`，则接口设计存在误导；若未来需要支持，则当前实现会掩盖传入 `true` 时的正确行为。此外，拼写错误 "suppport" 应为 "support"。
- **触发条件**: 用户传入 `fixedTriuMask = true`。
- **测试方案**: 传入 `fixedTriuMask = true`，确认返回 `ACLNN_ERR_PARAM_INVALID` 且日志信息正确。

## Bug 5: namespace 匿名空间内声明 extern 函数在 extern "C" 块中

- **位置**: 第 25-27 行、第 29-123 行、第 145-147 行
- **类型**: 编译/链接风险
- **严重程度**: 中
- **描述**: `extern "C"` 块包裹了整个实现，但匿名 namespace 内声明了两个 `extern` 函数（第 39-44 行）。匿名 namespace 具有内部链接属性，而 `extern` 声明期望外部链接，两者矛盾。在某些编译器上可能导致链接失败，找不到 `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 和 `aclnnInnerScaledMaskedSoftmaxV2` 的符号。
- **触发条件**: 使用严格的编译器或链接器时，匿名 namespace 中的 extern 声明可能无法正确解析外部符号。
- **测试方案**: 使用不同编译器（GCC、Clang）在严格模式下编译，检查是否有链接错误或警告。

---

# 汇总表

| 编号 | 位置(行号) | 类型 | 严重程度 | 简要描述 |
|------|-----------|------|---------|---------|
| 1 | 55-63 | 逻辑遗漏 | 高 | mask 数据类型未校验，MASK_DTYPE_SUPPORT_LIST 定义后未使用 |
| 2 | 104-106 | 日志错误 | 低 | 错误日志硬编码 4096，在 910_95 平台实际限制为 8192 |
| 3 | 74-121 | 逻辑遗漏 | 中 | 输出张量 y 的 shape 与 x 一致性未校验 |
| 4 | 131-135 | 接口缺陷 | 中 | fixedTriuMask 硬编码拒绝 true，拼写错误 |
| 5 | 29-44 | 编译/链接风险 | 中 | 匿名 namespace 内 extern 声明与 extern "C" 冲突 |
