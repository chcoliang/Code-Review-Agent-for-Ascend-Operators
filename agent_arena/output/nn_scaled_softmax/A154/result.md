# Code Review: aclnn_scaled_masked_softmax.cpp (A154)

## Bug 列表

### Bug 1: 错误日志中 D_LIMIT 硬编码与实际逻辑不一致

- **位置**: 第 106 行
- **类型**: 逻辑错误 / 错误信息不准确
- **严重程度**: 中
- **描述**: 当 SoC 为 ASCEND910_95 时，`dDimLimit` 被设为 `D_LIMIT_D = 8192`，但错误日志始终打印 `"Expected x and mask dim4 in range of (0, 4096]"`，未反映实际的动态限制值。这会误导用户，使其认为上限是 4096 而实际允许 8192。
- **触发条件**: 在 ASCEND910_95 平台上，当 `x` 的第4维大小在 (4096, 8192] 范围内时，校验通过但若出错则日志误导；若超过 8192 则报错信息中的范围不正确。
- **测试方案**: 在 ASCEND910_95 平台上构造 dim3=5000 的输入，验证校验通过；构造 dim3=9000 的输入，验证报错信息应显示 8192 而非 4096。

---

### Bug 2: CheckShape 未校验 x 与 y 的 shape 一致性

- **位置**: 第 113-122 行 (`CheckParams` 函数)
- **类型**: 校验缺失
- **严重程度**: 中
- **描述**: `CheckParams` 只校验了 `x` 和 `mask` 的 shape 关系，但从未校验输出张量 `y` 的 shape 是否与 `x` 一致。如果用户传入 shape 不匹配的 `y`，可能导致内存越界写入或计算结果错误。
- **触发条件**: 用户传入 `y` 的 shape 与 `x` 不同（例如 y 的某个维度更小），调用后续内核时产生未定义行为。
- **测试方案**: 构造 x shape=[2,4,8,16], y shape=[2,4,8,8]，调用 GetWorkspaceSize，预期应返回参数错误而非成功。

---

### Bug 3: `extern` 函数声明位于匿名 namespace 内部

- **位置**: 第 39-44 行
- **类型**: 编码规范 / 潜在链接错误
- **严重程度**: 低
- **描述**: `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 和 `aclnnInnerScaledMaskedSoftmaxV2` 的 `extern` 声明被放在匿名 namespace 内。匿名 namespace 赋予内部链接性，与 `extern`（外部链接）语义矛盾。虽然大多数编译器对 `extern` 声明不会真正应用匿名 namespace 的内部链接，但这属于未定义/实现定义行为，可能在某些编译器上导致链接失败。
- **触发条件**: 使用严格标准模式编译或某些特定编译器版本时可能出现链接错误。
- **测试方案**: 使用 `-std=c++17 -pedantic` 编译，观察是否有警告或链接错误。

---

### Bug 4: `fixedTriuMask` 参数被忽略，始终传 `false` 给内部函数

- **位置**: 第 136 行
- **类型**: 逻辑错误 / 功能限制未来扩展隐患
- **严重程度**: 低
- **描述**: 当 `fixedTriuMask=true` 时直接返回错误（第132-135行），当 `fixedTriuMask=false` 时调用内部函数但硬编码传入 `false`（第136行）。虽然当前逻辑正确（只支持 false），但硬编码 `false` 而非传入参数值 `fixedTriuMask` 本身是一个代码异味——如果将来去掉第132-135行的限制而忘记修改第136行，则 `fixedTriuMask=true` 的语义将永远丢失。
- **触发条件**: 未来维护时移除 `fixedTriuMask` 的限制检查但忘记修改硬编码值。
- **测试方案**: 代码审查确认；建议将第136行的 `false` 改为 `fixedTriuMask`。

---

### Bug 5: `extern "C"` 包裹了匿名 namespace 和非 C 兼容代码

- **位置**: 第 25-27 行, 第 146-148 行
- **类型**: 编码规范 / 潜在 ABI 问题
- **严重程度**: 低
- **描述**: `extern "C"` 块包裹了整个文件内容，包括匿名 namespace 中使用 C++ 特性（`std::initializer_list`、模板函数等）的代码。`extern "C"` 应仅包裹需要 C 链接的导出函数声明/定义。虽然匿名 namespace 中的内容不导出所以实际影响较小，但将 C++ 代码放在 `extern "C"` 块中属于不良实践，且对导出函数 `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 和 `aclnnScaledMaskedSoftmax` 会禁用 name mangling，若有重载可能冲突。
- **触发条件**: 如果需要增加同名重载函数或在纯 C 环境中链接时可能产生符号冲突。
- **测试方案**: 检查链接符号表，确认导出函数是否为 C 链接风格。

---

## 汇总表

| # | 位置 | 类型 | 严重程度 | 简述 |
|---|------|------|----------|------|
| 1 | 第106行 | 逻辑错误/错误信息不准确 | 中 | 错误日志硬编码4096，未适配动态dDimLimit |
| 2 | CheckParams | 校验缺失 | 中 | 未校验输出y与输入x的shape一致性 |
| 3 | 第39-44行 | 编码规范/链接问题 | 低 | extern声明放在匿名namespace内，语义矛盾 |
| 4 | 第136行 | 逻辑错误/代码异味 | 低 | fixedTriuMask硬编码false而非传参数值 |
| 5 | 第25-27行 | 编码规范/ABI问题 | 低 | extern "C"不当包裹C++代码 |
