# A153 代码审查报告

## Bug 1: workspaceSize 参数传入 nullptr 导致调用者无法获取所需空间大小

- **位置**: 第 136 行
- **类型**: 空指针参数错误
- **严重程度**: 高
- **描述**: `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 的第 6 个参数应传入调用者提供的 `workspaceSize` 指针以便回写所需的 workspace 大小，但此处传入了 `nullptr`。这导致调用者永远无法获得正确的 workspaceSize 值，后续分配 workspace 时将使用未初始化或错误的大小，导致内存不足或越界。
- **触发条件**: 每次调用 `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 时必然触发。调用者获取到的 workspaceSize 保持未初始化状态（或初始值 0），后续执行 `aclnnScaledMaskedSoftmax` 时会因 workspace 不足而崩溃或产生错误结果。
- **测试方案**: 
  1. 调用 `aclnnScaledMaskedSoftmaxGetWorkspaceSize`，检查返回后 `workspaceSize` 的值是否为合理正值。
  2. 完整调用 GetWorkspaceSize -> 分配 workspace -> 执行算子的流程，验证是否崩溃。

## Bug 2: 错误日志中 DIM_3 范围描述不准确

- **位置**: 第 106 行
- **类型**: 日志信息错误
- **严重程度**: 低
- **描述**: 错误信息固定写为 `"Expected x and mask dim4 in range of (0, 4096]."`，但当平台为 ASCEND910_95 时，实际限制为 8192。错误提示会误导用户。
- **触发条件**: 在 ASCEND910_95 平台上，传入 DIM_3 超过 8192 的 tensor 时，提示的上限仍为 4096，与实际不符。
- **测试方案**: 在 ASCEND910_95 平台上传入 DIM_3=9000 的 tensor，验证错误信息是否准确反映 8192 的上限。

---

## 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 136 | 空指针参数错误 | 高 | 将 nullptr 传给 workspaceSize 参数，调用者无法获取所需 workspace 大小 |
| 2 | 106 | 日志信息错误 | 低 | 错误信息未根据平台动态显示实际的 DIM_3 上限值 |
