# Code Review: aclnn_scaled_masked_softmax.cpp (A147)

## Bug 1: 错误日志中 DIM_3 上限信息与实际逻辑不一致

- **位置**: 第 101 行
- **类型**: 错误信息不准确 (Misleading Error Message)
- **严重程度**: 中
- **描述**: 当平台为 ASCEND910_95 时，`dDimLimit` 被设置为 8192，但错误日志硬编码为 `"Expected x and mask dim4 in range of (0, 4096]."` 未反映实际的 8192 上限。用户在 910_95 平台上输入 dim3 为 5000 时，会看到错误提示说上限是 4096，实际上限是 8192，导致误导。
- **触发条件**: 在 ASCEND910_95 平台上，输入 tensor x 的第 4 维大小超过 8192 时触发该错误日志。
- **测试方案**: 在 ASCEND910_95 平台上构造 x shape 为 [1,1,1,8193]，检查返回的错误日志是否正确描述限制为 8192。

## Bug 2: mask 与 x 的 DIM_2 和 DIM_3 维度缺少广播/一致性校验

- **位置**: 第 75-106 行 (`CheckShape` 函数)
- **类型**: 校验缺失 (Missing Validation)
- **严重程度**: 高
- **描述**: `CheckShape` 函数仅校验了 x 和 mask 在 DIM_0 和 DIM_1 维度的广播关系，但完全未校验 DIM_2 和 DIM_3 维度是否一致或满足广播条件。如果 mask 的后两维与 x 不匹配且不为 1，则可能导致内核执行时内存越界访问或计算结果错误。
- **触发条件**: 传入 x shape=[2,4,8,16], mask shape=[2,4,3,7]（后两维不匹配且不为 1）。
- **测试方案**: 构造 x=[B,H,S,D] 与 mask=[B,H,S',D']（S'!=S, D'!=D, 且都不为1），调用 GetWorkspaceSize，期望返回 ACLNN_ERR_PARAM_INVALID 但实际不会。

## Bug 3: x 与 y 的 shape 一致性未校验

- **位置**: 第 108-118 行 (`CheckParams` 函数)
- **类型**: 校验缺失 (Missing Validation)
- **严重程度**: 高
- **描述**: 输出 tensor y 仅校验了 dtype 与 x 一致以及维度数为 4，但从未校验 y 的 shape 是否与 x 的 shape 相同。若 y 的 shape 小于 x，内核写入时将发生内存越界。
- **触发条件**: 传入 x shape=[2,4,8,16], y shape=[1,1,1,1]（dtype 相同，维度数为 4），将通过所有现有检查。
- **测试方案**: 构造 x=[2,4,8,16], y=[2,4,8,1]，调用算子，验证是否返回参数错误（预期应该报错但当前不会）。

## Bug 4: namespace 匿名空间内声明 extern 函数

- **位置**: 第 39-44 行
- **类型**: 链接问题 (Linkage Issue)
- **严重程度**: 中
- **描述**: `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 和 `aclnnInnerScaledMaskedSoftmaxV2` 被声明在匿名命名空间内部（第 29 行开始），匿名命名空间内的符号具有内部链接属性。将 `extern` 声明放在匿名命名空间中，在某些编译器下可能导致链接失败，因为编译器会为其赋予内部链接，而定义在其他翻译单元中具有外部链接。不同编译器对此行为不一致。
- **触发条件**: 使用严格遵循标准的编译器编译时可能出现链接错误。
- **测试方案**: 使用 `-pedantic` 编译选项编译，检查是否有链接警告或错误。

## Bug 5: fixedTriuMask 参数校验位置不当，可能空指针解引用

- **位置**: 第 127-130 行
- **类型**: 逻辑问题 (Logic Issue)
- **严重程度**: 低
- **描述**: `fixedTriuMask` 的校验放在 `CheckParams` 之后，这本身没有直接 bug，但如果未来 `fixedTriuMask=true` 时需要跳过某些参数校验（如 mask 可为 null），当前顺序会导致先对 mask 做空指针检查并报错，而非优先报告不支持 fixedTriuMask=true。这是设计问题，当前场景下影响有限。
- **触发条件**: 当 fixedTriuMask=true 且 mask=nullptr 时，报错信息为空指针而非不支持该参数。
- **测试方案**: 调用时传入 fixedTriuMask=true, mask=nullptr，观察错误码是否为预期的 ACLNN_ERR_PARAM_INVALID。

## Bug 6: DIM_3 校验仅检查了 x，未检查 mask 的对应维度上限

- **位置**: 第 100-103 行
- **类型**: 校验不完整 (Incomplete Validation)
- **严重程度**: 低
- **描述**: 错误信息写的是 "Expected x and mask dim4 in range of (0, 4096]"，暗示 mask 的 dim4 也应该被校验，但实际代码只检查了 `x->GetViewShape().GetDim(DIM_3)`，未检查 mask 对应维度。如果 mask 的 DIM_3 超限，可能导致内核处理异常。
- **触发条件**: x shape=[1,1,1,32], mask shape=[1,1,1,8192]（mask 的 DIM_3 超限但 x 的 DIM_3 未超限），在非 910_95 平台上。
- **测试方案**: 构造 x dim3=32, mask dim3=5000，验证是否能通过校验。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 101 | 错误信息不准确 | 中 | 错误日志硬编码4096，未适配8192场景 |
| 2 | 75-106 | 校验缺失 | 高 | mask的DIM_2和DIM_3未校验广播关系 |
| 3 | 108-118 | 校验缺失 | 高 | 输出y的shape未与x做一致性校验 |
| 4 | 39-44 | 链接问题 | 中 | extern声明在匿名namespace中，可能链接异常 |
| 5 | 127-130 | 逻辑问题 | 低 | fixedTriuMask校验顺序可致误导性报错 |
| 6 | 100-103 | 校验不完整 | 低 | 仅校验x的DIM_3上限，未校验mask |
