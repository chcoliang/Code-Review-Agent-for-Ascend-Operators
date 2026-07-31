# Code Review: aclnn_apply_adam_w.cpp (A158)

## Bug List

### Bug 1: CheckNotNull 失败时返回 ACLNN_SUCCESS 而非错误码

- **位置**: 第 138-139 行, `CheckParams` 函数
- **类型**: 逻辑错误 / 错误处理缺陷
- **严重程度**: 严重 (Critical)
- **描述**: `CHECK_RET(CheckNotNull(...), ACLNN_SUCCESS)` 当 `CheckNotNull` 返回 `false` 时，`CHECK_RET` 宏会返回第二个参数作为错误码。此处第二个参数为 `ACLNN_SUCCESS`，这意味着空指针校验失败时函数仍然返回成功，后续代码将解引用空指针导致崩溃。应改为 `ACLNN_ERR_PARAM_NULLPTR` 或类似的错误码。
- **触发条件**: 任何必填 tensor 参数传入 nullptr 时触发。
- **测试方案**: 调用 `aclnnApplyAdamWGetWorkspaceSize` 时将 `varRef` 设为 nullptr，验证返回值是否为错误码（非 ACLNN_SUCCESS）。

### Bug 2: 标量 tensor 未做 Contiguous 转换

- **位置**: 第 198-200 行, `aclnnApplyAdamWGetWorkspaceSize` 函数
- **类型**: 数据正确性 / 内存访问错误
- **严重程度**: 高 (High)
- **描述**: `beta1Power`、`beta2Power`、`lr`、`weightDecay`、`beta1`、`beta2`、`eps` 这些标量 tensor 在传入 `l0op::ApplyAdamW` 之前未经过 `l0op::Contiguous` 转换。如果这些 tensor 的存储格式为非连续（例如经过 slice/transpose 后），则内核读取数据时会发生地址偏移错误，导致计算结果错误或非法内存访问。
- **触发条件**: 传入的标量 tensor 为非连续 tensor（如 `tensor.as_strided([1], [2])`）。
- **测试方案**: 构造 stride != 1 的标量 tensor 作为 `beta1` 传入，对比输出结果与期望值。

### Bug 3: amsgrad=true 时缺少 maxGradNorm 的 ViewCopy 回写

- **位置**: 第 205-216 行, `aclnnApplyAdamWGetWorkspaceSize` 函数
- **类型**: 逻辑遗漏 / 数据丢失
- **严重程度**: 高 (High)
- **描述**: 当 `amsgrad=true` 且 `maxGradNormOptional` 非空时，该 tensor 会被 `Contiguous` 转换后传入 `ApplyAdamW` 参与计算并被更新。但后续的 ViewCopy 回写逻辑仅处理了 `varRef`、`mRef`、`vRef`，遗漏了 `maxGradNormOptional`。如果 `maxGradNormOptional` 本身是非连续 tensor，则计算结果无法写回原始 tensor，导致 amsgrad 状态丢失。
- **触发条件**: `amsgrad=true`，且传入的 `maxGradNormOptional` 为非连续 tensor。
- **测试方案**: 构造非连续的 maxGradNorm tensor，执行 amsgrad=true 的 AdamW，检查 maxGradNorm 是否被正确更新。

### Bug 4: 未校验 workspaceSize 和 executor 输出指针是否为空

- **位置**: 第 151-156 行, `aclnnApplyAdamWGetWorkspaceSize` 函数
- **类型**: 参数校验缺失 / 空指针解引用
- **严重程度**: 中 (Medium)
- **描述**: 函数参数 `workspaceSize`（第 171/219 行解引用）和 `executor`（第 172/220 行通过 ReleaseTo 使用）作为输出参数，未在函数入口进行 nullptr 校验。若调用方传入 nullptr，会直接导致段错误。
- **触发条件**: 调用方传入 `workspaceSize=nullptr` 或 `executor=nullptr`。
- **测试方案**: 分别传入 nullptr 给 workspaceSize 和 executor，验证函数不崩溃并返回错误码。

### Bug 5: aclnnApplyAdamW 缺少参数校验

- **位置**: 第 224-228 行, `aclnnApplyAdamW` 函数
- **类型**: 参数校验缺失
- **严重程度**: 中 (Medium)
- **描述**: 执行阶段函数 `aclnnApplyAdamW` 未对 `executor`、`stream` 进行空指针校验，也未验证 `workspace` 在 `workspaceSize > 0` 时是否为非空。直接传给 `CommonOpExecutorRun` 可能导致内部崩溃且难以定位原因。
- **触发条件**: 调用方在 GetWorkspaceSize 后错误地传入 nullptr executor 或 stream。
- **测试方案**: 传入 executor=nullptr 调用 aclnnApplyAdamW，验证行为（应返回错误码而非崩溃）。

### Bug 6: CheckShape 中标量 tensor 元素数校验失败时无错误日志

- **位置**: 第 125-128 行, `CheckShape` 函数
- **类型**: 可维护性 / 调试困难
- **严重程度**: 低 (Low)
- **描述**: 当 `beta1Power`、`beta2Power`、`lr` 等标量 tensor 的 `Numel() != 1` 时，函数直接返回 `false`，没有输出任何错误日志信息。其他校验使用了 `OP_CHECK_SHAPE_NOT_EQUAL` 宏（通常会记录日志），但此处手工校验遗漏了日志输出，排查问题时难以定位是哪个参数 shape 不合规。
- **触发条件**: 任何标量 tensor 的元素个数不为 1。
- **测试方案**: 传入元素数 > 1 的 lr tensor，检查日志中是否有明确的参数错误提示。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简述 |
|------|------|----------|----------|------|
| 1 | L139 | 错误处理 | Critical | 空指针校验失败返回 SUCCESS |
| 2 | L198-200 | 数据正确性 | High | 标量 tensor 未做 Contiguous |
| 3 | L205-216 | 逻辑遗漏 | High | maxGradNorm 缺少 ViewCopy 回写 |
| 4 | L151-156 | 参数校验 | Medium | 输出指针未校验 nullptr |
| 5 | L224-228 | 参数校验 | Medium | 执行阶段无参数校验 |
| 6 | L125-128 | 可维护性 | Low | 标量 shape 校验无日志 |
