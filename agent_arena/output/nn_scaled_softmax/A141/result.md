# Code Review: aclnn_scaled_masked_softmax.cpp (A141)

## Bug 1: 错误日志中 DIM_3 范围描述与实际逻辑不一致

- **位置**: 第 105 行
- **类型**: 日志信息错误
- **严重程度**: 中
- **描述**: 错误日志硬编码为 `"Expected x and mask dim4 in range of (0, 4096]."`, 但实际上当平台为 ASCEND910_95 时, `dDimLimit` 为 8192。日志信息会误导用户，使其认为上限始终是 4096。
- **触发条件**: 在 ASCEND910_95 平台上，当 `x` 的第 4 维大于 8192 时触发该错误分支，但用户看到的提示为 4096，造成困惑。
- **测试方案**: 在 ASCEND910_95 环境下，传入第 4 维为 5000 的张量（合法值），观察是否被错误拒绝（不会），再传入 9000 的张量触发错误，检查错误信息是否正确反映 8192 的限制。

## Bug 2: namespace 内声明 extern 函数与 extern "C" 作用域冲突

- **位置**: 第 25-27 行, 第 29 行, 第 39-44 行, 第 123 行, 第 145-147 行
- **类型**: 链接/编译错误
- **严重程度**: 高
- **描述**: `extern "C"` 块从第 26 行开始，匿名 namespace 在第 29 行开始。匿名 namespace 内第 39-44 行声明了两个 `extern` 函数 (`aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 和 `aclnnInnerScaledMaskedSoftmaxV2`)。匿名 namespace 赋予符号内部链接(internal linkage)，与 `extern` 声明矛盾——extern 要求外部链接。这会导致链接时找不到这两个函数的定义，或者编译器报错/警告。
- **触发条件**: 编译该文件并链接时，链接器可能无法解析匿名 namespace 内的 extern 符号。
- **测试方案**: 编译该文件并链接完整项目，检查是否出现 undefined reference 或 linkage 相关的编译器警告。将这两个 extern 声明移到匿名 namespace 外部进行修复验证。

## Bug 3: 缺少输出张量 y 与输入张量 x 的 shape 一致性校验

- **位置**: 第 74 行 (`CheckShape` 函数)
- **类型**: 校验缺失
- **严重程度**: 中
- **描述**: `CheckShape` 函数只校验了 `x` 和 `mask` 之间的形状关系，但未校验输出张量 `y` 的形状是否与 `x` 一致。如果用户传入 shape 不匹配的 `y`，可能导致内存越界写入或计算结果错误。
- **触发条件**: 用户传入一个 shape 与 `x` 不同的 `y` 张量。
- **测试方案**: 构造 x shape=[2,4,8,16]，y shape=[1,1,1,1]，调用该接口，观察是否报错或产生内存越界。

## Bug 4: 缺少输入输出 dtype 一致性校验

- **位置**: 第 55-63 行 (`CheckDtypeValid` 函数)
- **类型**: 校验缺失
- **严重程度**: 中
- **描述**: `CheckDtypeValid` 分别检查了 x 和 y 的 dtype 是否在支持列表中，但未校验 x 与 y 的 dtype 是否相同。Scaled Softmax 的输出 dtype 通常应与输入一致，若不一致可能导致精度问题或未定义行为。
- **触发条件**: 用户传入 x 为 FP16, y 为 BF16（两者都在支持列表中但不同）。
- **测试方案**: 构造 x dtype=FP16, y dtype=BF16，调用接口并观察计算结果是否正确或是否报错。

## Bug 5: fixedTriuMask 参数被忽略，硬编码为 false

- **位置**: 第 131-135 行
- **类型**: 逻辑错误/功能缺陷
- **严重程度**: 低
- **描述**: 当 `fixedTriuMask` 为 `true` 时直接返回错误，当为 `false` 时调用内部函数但硬编码传入 `false`。虽然当前实现可能是故意限制，但接口对外暴露了该参数却完全不支持其功能，且错误提示有拼写错误 "suppport"（多了一个 p）。这属于接口设计问题，如果后续要支持该功能需要修改。
- **触发条件**: 用户传入 `fixedTriuMask = true`。
- **测试方案**: 调用接口时设置 `fixedTriuMask = true`，确认返回 `ACLNN_ERR_PARAM_INVALID` 错误。

## Bug 6: scale 参数未做有效性校验

- **位置**: 第 125-137 行 (`aclnnScaledMaskedSoftmaxGetWorkspaceSize` 函数)
- **类型**: 校验缺失
- **严重程度**: 低
- **描述**: `scale` 参数为 double 类型，未校验特殊值（如 NaN、Inf、负数、0）。若 scale 为 NaN 或 Inf，可能导致后续计算异常。
- **触发条件**: 用户传入 `scale = NaN` 或 `scale = +Inf`。
- **测试方案**: 传入 scale 为 NaN/Inf/0/负数，观察是否正常报错或产生异常计算结果。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 105 | 日志信息错误 | 中 | 错误提示硬编码4096，未适配8192场景 |
| 2 | 29, 39-44 | 链接/编译错误 | 高 | 匿名namespace内extern声明导致链接冲突 |
| 3 | 74 | 校验缺失 | 中 | 未校验输出y与输入x的shape一致性 |
| 4 | 55-63 | 校验缺失 | 中 | 未校验x与y的dtype是否相同 |
| 5 | 131-135 | 逻辑/拼写错误 | 低 | fixedTriuMask不支持且提示有拼写错误 |
| 6 | 125-137 | 校验缺失 | 低 | scale参数未校验NaN/Inf等特殊值 |
