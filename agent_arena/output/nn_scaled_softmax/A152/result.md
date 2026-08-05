# Code Review: aclnn_scaled_masked_softmax.cpp (A152)

## Bug 列表

### Bug 1: 错误日志中 D_LIMIT 范围描述硬编码，未反映动态限制值

- **位置**: 第 106 行
- **类型**: 逻辑错误 / 信息不一致
- **严重程度**: 中
- **描述**: 当 SoC 版本为 ASCEND910_95 时，`dDimLimit` 被设为 8192，但错误日志始终输出 `"Expected x and mask dim4 in range of (0, 4096]."` 。在 910_95 平台上，实际允许范围是 (0, 8192]，但报错信息仍显示 4096，这会误导用户排查问题。
- **触发条件**: 在 ASCEND910_95 平台上，当 `x` 的第4维大小在 (4096, 8192] 范围内时不会触发错误（正常通过），但当大小 > 8192 时触发的错误信息内容不准确；同样在非910_95平台，dim4 > 4096 时错误信息正确。
- **测试方案**: 在 ASCEND910_95 环境下构造 dim4=9000 的输入，验证报错信息应显示 8192 而非 4096。

---

### Bug 2: x 与 y 的 shape 一致性未校验

- **位置**: `CheckParams` 函数（第 113-123 行），缺少 x 与 y 的 shape 比较
- **类型**: 缺失校验
- **严重程度**: 高
- **描述**: 输出 tensor `y` 的 shape 应与输入 `x` 完全相同（softmax 不改变形状），但代码中只校验了 x 与 mask 的 shape 关系，从未验证 `y` 的 shape 是否与 `x` 一致。如果用户传入 shape 不匹配的 `y`，将导致内存越界写入或结果错误。
- **触发条件**: 用户传入的 `y` tensor 的 shape 与 `x` 不一致时。
- **测试方案**: 构造 x shape=[2,4,8,16]，y shape=[2,4,8,32]，调用接口应返回参数错误而非崩溃。

---

### Bug 3: `extern "C"` 内部包含了匿名 namespace 和 C++ 特性代码

- **位置**: 第 25-27 行 `#ifdef __cplusplus extern "C" {` 与第 29-124 行的匿名 namespace
- **类型**: 代码结构/链接错误
- **严重程度**: 中
- **描述**: `extern "C"` 块旨在使用 C 链接规范导出函数，但块内包含了匿名 namespace（C++ 特性）、`std::initializer_list`、模板等纯 C++ 构造。虽然编译器通常可以处理（因为有 `#ifdef __cplusplus` 保护），但匿名 namespace 中的 `extern` 声明（第 39-44 行）在 `extern "C"` 内声明外部 C++ 函数，可能导致链接符号修饰（name mangling）不匹配，找不到正确的符号。
- **触发条件**: 链接阶段，如果 `aclnnInnerScaledMaskedSoftmaxGetWorkspaceSize` 和 `aclnnInnerScaledMaskedSoftmaxV2` 实际以 C++ mangling 导出，则在 `extern "C"` 内用 C linkage 声明会导致链接失败。
- **测试方案**: 完整编译链接该模块，检查是否有 undefined symbol 错误。

---

### Bug 4: `fixedTriuMask` 参数校验后硬编码传 false，忽略用户意图

- **位置**: 第 132-136 行
- **类型**: 功能限制 / 潜在逻辑问题
- **严重程度**: 低
- **描述**: 当 `fixedTriuMask` 为 true 时直接报错返回，为 false 时调用内部接口也硬编码传 `false`。当前实现仅支持 `fixedTriuMask=false`，但错误信息拼写为 "suppport"（多一个 p），且没有将用户传入的值透传给内部函数。如果将来支持 true，需要修改此处。此处拼写错误会影响用户体验。
- **触发条件**: 用户传入 `fixedTriuMask=true` 时触发错误信息，信息中有拼写错误。
- **测试方案**: 传入 `fixedTriuMask=true`，检查返回的错误信息是否拼写正确（"support" vs "suppport"）。

---

### Bug 5: `aclnnScaledMaskedSoftmax` 执行函数缺少参数空指针校验

- **位置**: 第 140-144 行
- **类型**: 缺失校验
- **严重程度**: 中
- **描述**: `aclnnScaledMaskedSoftmax` 函数直接将 `workspace`、`executor`、`stream` 转发给内部函数，没有做任何空指针或有效性检查。如果 `executor` 或 `stream` 为 nullptr，可能导致段错误。
- **触发条件**: 用户未正确调用 GetWorkspaceSize 就直接调用执行函数，或传入 nullptr executor/stream。
- **测试方案**: 传入 executor=nullptr 调用 `aclnnScaledMaskedSoftmax`，期望返回错误码而非崩溃。

---

### Bug 6: CheckShape 中未校验 mask 的 DIM_2/DIM_3 广播条件的完整性

- **位置**: 第 94-98 行
- **类型**: 逻辑不完整
- **严重程度**: 低
- **描述**: 对于 DIM_0 和 DIM_1，代码允许 mask 为 1（广播），但对 DIM_2 和 DIM_3 要求严格相等。这是设计意图，但与函数注释/日志不一致——日志说 "x and mask should be same in dim3 and dim4"，实际检查的是索引 DIM_2 和 DIM_3（即第3维和第4维，0-based），而日志使用 1-based "dim3 and dim4"。如果设计确实要求 DIM_2/DIM_3 严格相等则无问题，只是命名可能造成理解混淆。
- **触发条件**: 当 x 和 mask 在 DIM_2 或 DIM_3 不等时触发。
- **测试方案**: 确认日志与实际维度含义一致即可。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第106行 | 逻辑错误/信息不一致 | 中 | 错误日志硬编码4096，未反映910_95平台的8192限制 |
| 2 | CheckParams函数 | 缺失校验 | 高 | 未校验输出y与输入x的shape一致性 |
| 3 | 第25-27行 extern "C" | 代码结构/链接 | 中 | extern "C"内包含C++ namespace和extern声明，可能链接失败 |
| 4 | 第132-136行 | 拼写错误/功能限制 | 低 | "suppport"拼写错误，硬编码false |
| 5 | 第140-144行 | 缺失校验 | 中 | 执行函数缺少空指针校验 |
| 6 | 第94-98行 | 日志描述不准确 | 低 | 日志维度命名与0-based索引不一致 |
