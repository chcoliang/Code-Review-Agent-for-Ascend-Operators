# Code Review: aclnn_scaled_masked_softmax.cpp (A142)

## Bug List

### Bug 1: Mask dtype 未校验 (MASK_DTYPE_SUPPORT_LIST 定义但未使用)

- **位置**: 第 55-63 行, `CheckDtypeValid` 函数
- **类型**: 逻辑缺陷 / 输入校验遗漏
- **严重程度**: 高
- **描述**: `MASK_DTYPE_SUPPORT_LIST`(第 37 行)定义了 mask 仅支持 `DT_BOOL` 类型，但 `CheckDtypeValid` 函数中从未对 `mask` 参数进行 dtype 校验。函数接收了 `mask` 参数却完全忽略了对它的检查。如果用户传入非 BOOL 类型的 mask tensor，将绕过校验直接进入内部计算，可能导致计算结果错误或内存访问异常。
- **触发条件**: 传入 dtype 为 DT_FLOAT、DT_INT32 等非 DT_BOOL 类型的 mask tensor。
- **测试方案**: 构造一个 dtype 为 DT_FLOAT16 的 mask tensor，调用 `aclnnScaledMaskedSoftmaxGetWorkspaceSize`，预期应返回 `ACLNN_ERR_PARAM_INVALID`，但实际会返回 `ACLNN_SUCCESS` 或在后续执行中崩溃。

---

### Bug 2: 错误信息与实际限制不一致

- **位置**: 第 105 行, `CheckShape` 函数
- **类型**: 错误信息不准确
- **严重程度**: 中
- **描述**: 当运行在 `ASCEND910_95` 平台时，`dDimLimit` 实际为 8192，但错误日志仍硬编码输出 `"Expected x and mask dim4 in range of (0, 4096]."`. 这会误导用户，使其认为上限是 4096 而非 8192，增加调试难度。
- **触发条件**: 在 ASCEND910_95 平台上，传入 dim3 大小超过 8192 的 tensor 触发该错误分支。
- **测试方案**: 在 ASCEND910_95 平台上传入 dim3=8192+1 的 x tensor，检查返回的错误信息是否正确反映实际限制值 8192。

---

### Bug 3: 缺少输出 tensor y 与输入 x 的 shape 一致性校验

- **位置**: 第 74-110 行, `CheckShape` 函数 及 第 112-121 行 `CheckParams` 函数
- **类型**: 输入校验遗漏
- **严重程度**: 高
- **描述**: `CheckShape` 仅校验了 x 与 mask 的 shape 关系，但从未校验输出 tensor y 的 shape 是否与 x 一致。Scaled Masked Softmax 的输出 shape 应当与输入 x 完全相同。如果 y 的 shape 与 x 不匹配（如 y 分配的空间更小），可能导致内存越界写入。
- **触发条件**: 传入一个 shape 与 x 不同的 y tensor（例如 y 的某个维度更小）。
- **测试方案**: 构造 x shape 为 [2,4,8,16], y shape 为 [2,4,8,8] 的 tensor 对，调用接口，预期应返回参数错误，但实际会透传到内部算子可能导致内存越界。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | L55-63 `CheckDtypeValid` | 输入校验遗漏 | 高 | mask 的 dtype 未校验，MASK_DTYPE_SUPPORT_LIST 定义未使用 |
| 2 | L105 `CheckShape` | 错误信息不准确 | 中 | 错误日志硬编码 4096 但实际限制可能为 8192 |
| 3 | L74-110 `CheckShape` | 输入校验遗漏 | 高 | 缺少 y 与 x 的 shape 一致性校验，可能越界写入 |
