# Code Review: aclnn_scaled_masked_softmax.cpp (A143)

## Bug 1: 错误日志中 DIM_3 范围描述与实际逻辑不一致

- **位置**: 第 105 行
- **类型**: 日志信息错误
- **严重程度**: 中
- **描述**: 当平台为 ASCEND910_95 时，`dDimLimit` 被设置为 8192，但错误日志硬编码为 `"Expected x and mask dim4 in range of (0, 4096]."` 。当实际限制为 8192 时，用户看到的错误提示与实际约束不符，会造成误导。
- **触发条件**: 在 ASCEND910_95 平台上，当 `x` 的第4维度值大于 8192 时触发该错误分支，但日志提示限制为 4096。
- **测试方案**: 在 ASCEND910_95 平台上传入 DIM_3 = 5000 的 tensor，检查返回错误信息是否正确反映 8192 的上限。

## Bug 2: namespace 内声明 extern 函数位于 `extern "C"` 块中

- **位置**: 第 25-27 行, 第 29 行, 第 39-44 行
- **类型**: 链接/编译错误
- **严重程度**: 高
- **描述**: `extern "C"` 块从第 26 行开始，匿名 namespace 在第 29 行开始，其中第 39-44 行声明了两个 `extern` 函数（`aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 和 `aclnnInnerScaledMaskedSoftmaxV2`）。这些函数被放置在匿名 namespace 内部，具有内部链接性（internal linkage），但又用 `extern` 声明期望外部链接。匿名 namespace 会使这些声明具有内部链接性，导致无法正确链接到外部定义的这两个函数实现。
- **触发条件**: 编译链接时，如果这两个函数的定义在其他翻译单元中，链接器将找不到它们的符号。
- **测试方案**: 将该文件单独编译并与提供这两个函数实现的库进行链接，观察是否出现 undefined reference 链接错误。

## Bug 3: fixedTriuMask 参数被忽略，硬编码为 false

- **位置**: 第 131-135 行
- **类型**: 逻辑错误/功能缺陷
- **严重程度**: 中
- **描述**: 函数 `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 接受 `fixedTriuMask` 参数，但当其为 `true` 时直接返回错误，当为 `false` 时传给内部函数的也是硬编码的 `false`（第135行）而非参数本身。虽然当前只支持 `false`，但传入硬编码值而非参数值意味着未来扩展时容易遗漏修改。这是一个潜在的维护性问题，且接口设计上对用户产生误导（声明支持该参数但实际不支持）。
- **触发条件**: 用户传入 `fixedTriuMask = true` 时得到错误而非预期的上三角 mask 功能。
- **测试方案**: 调用时传入 `fixedTriuMask = true`，验证返回 `ACLNN_ERR_PARAM_INVALID`；确认错误信息拼写正确（注意当前 "suppport" 是拼写错误）。

## Bug 4: 错误日志拼写错误 "suppport"

- **位置**: 第 132 行
- **类型**: 拼写错误
- **严重程度**: 低
- **描述**: 日志消息 `"the param fixedTriuMask only suppport false."` 中 "suppport" 应为 "support"。
- **触发条件**: 当 `fixedTriuMask = true` 时触发该日志输出。
- **测试方案**: 传入 `fixedTriuMask = true`，检查日志输出中的拼写。

## Bug 5: CheckShape 未校验 x 与 y 的 shape 一致性

- **位置**: 第 74-110 行 / 第 112-121 行
- **类型**: 校验遗漏
- **严重程度**: 中
- **描述**: `CheckParams` 流程中只检查了 x 与 mask 的 shape 关系，但未检查输出 tensor `y` 的 shape 是否与 `x` 一致。如果 `y` 的 shape 与 `x` 不匹配，可能导致运行时内存越界写入。
- **触发条件**: 用户传入 shape 不匹配的 `y` tensor（例如 `y` 的某一维度小于 `x`），内部算子写入时可能越界。
- **测试方案**: 传入 `x` shape 为 [2,4,8,64]，`y` shape 为 [2,4,8,32]，观察是否出现内存错误或是否被正确拦截。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 105 | 日志信息错误 | 中 | DIM_3 范围日志硬编码4096，未适配8192场景 |
| 2 | 39-44 (在匿名namespace内) | 链接错误 | 高 | extern函数声明在匿名namespace内，内部链接性与extern矛盾 |
| 3 | 131-135 | 逻辑错误 | 中 | fixedTriuMask参数硬编码false传递给内部函数 |
| 4 | 132 | 拼写错误 | 低 | "suppport" 应为 "support" |
| 5 | 74-121 | 校验遗漏 | 中 | 未校验输出tensor y与输入x的shape一致性 |
