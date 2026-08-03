# Code Review: aclnn_apply_adam_w.cpp (A165)

## Bug List

### Bug 1: grad tensor shape未校验

- **位置**: `CheckShape` 函数, 第115-129行
- **类型**: 参数校验缺失
- **严重程度**: 高
- **描述**: `CheckShape` 函数校验了 `mRef`、`vRef`、`maxGradNormOptional` 与 `varRef` 的 shape 一致性，但遗漏了对 `grad` tensor 的 shape 校验。AdamW 算法要求 grad 的 shape 必须与 var 一致，否则会导致计算越界或结果错误。
- **触发条件**: 用户传入一个与 `varRef` shape 不同的 `grad` tensor（例如 varRef shape 为 [1024, 512]，grad shape 为 [512, 1024]），函数不会报错，直接进入计算逻辑。
- **测试方案**: 构造 varRef shape=[4,4]、grad shape=[2,8] 的 tensor 调用接口，预期返回 `ACLNN_ERR_PARAM_INVALID`，实际会进入计算导致未定义行为。

---

### Bug 2: workspaceSize 和 executor 输出指针未做空指针校验

- **位置**: `aclnnApplyAdamWGetWorkspaceSize` 函数, 第150-221行（使用点在第218-219行）
- **类型**: 参数校验缺失
- **严重程度**: 高
- **描述**: 函数入参 `uint64_t* workspaceSize` 和 `aclOpExecutor** executor` 为输出参数指针，但函数未对其进行空指针检查。当用户传入 nullptr 时，第170行 `*workspaceSize = 0` 或第218行 `*workspaceSize = uniqueExecutor->GetWorkspaceSize()` 会导致空指针解引用崩溃。
- **触发条件**: 调用 `aclnnApplyAdamWGetWorkspaceSize(...)` 时，将 `workspaceSize` 或 `executor` 参数传为 `nullptr`。
- **测试方案**: 传入合法 tensor 参数但 workspaceSize=nullptr，预期返回错误码，实际程序段错误崩溃。

---

### Bug 3: 标量tensor（beta1Power等）未做Contiguous转换

- **位置**: `aclnnApplyAdamWGetWorkspaceSize` 函数, 第197-199行
- **类型**: 边界条件处理不完整
- **严重程度**: 低
- **描述**: `beta1Power`、`beta2Power`、`lr`、`weightDecay`、`beta1`、`beta2`、`eps` 这些标量 tensor 在传给 `l0op::ApplyAdamW` 之前未经过 `l0op::Contiguous` 转换。虽然标量 tensor 通常是连续的，但如果用户通过 view/slice 等方式构造了非连续的标量 tensor（numel=1 但 stride 异常），可能导致内核读取错误地址。
- **触发条件**: 对一个多元素 tensor 做 slice 操作得到 numel=1 的非连续视图，作为 beta1Power 等参数传入。
- **测试方案**: 构造一个 shape=[1] 但 storage_offset 非零、stride 异常的 tensor 作为 lr 传入，观察计算结果是否正确。

---

### Bug 4: CheckShape 中标量参数校验失败时缺少错误日志

- **位置**: `CheckShape` 函数, 第124-127行
- **类型**: 错误处理不完善
- **严重程度**: 低
- **描述**: 当 `beta1Power`、`beta2Power`、`lr`、`weightDecay`、`beta1`、`beta2`、`eps` 的 Numel 不为1时，函数直接 `return false` 而没有打印任何错误日志。其他校验都使用了 `OP_CHECK_*` 宏（内部会记录错误信息），但此处的手动校验无法帮助用户定位问题。
- **触发条件**: 传入 numel != 1 的 beta1 tensor 时，接口返回 `ACLNN_ERR_PARAM_INVALID` 但无日志说明哪个参数有问题。
- **测试方案**: 传入 shape=[2] 的 beta1 tensor，检查是否有明确的错误日志输出指示是哪个参数校验失败。

---

### Bug 5: amsgrad=true 时 maxGradNormOptional 结果未回写

- **位置**: `aclnnApplyAdamWGetWorkspaceSize` 函数, 第204-215行
- **类型**: 逻辑缺陷
- **严重程度**: 高
- **描述**: 当 `amsgrad=true` 时，`maxGradNormOptional` 是一个需要更新的 inout 参数（存储历史最大梯度平方的移动平均）。代码对 `varRef`、`mRef`、`vRef` 做了非连续场景的 ViewCopy 回写，但遗漏了 `maxGradNormOptional` 的回写处理。如果 `maxGradNormOptional` 是非连续 tensor，其计算结果将丢失。
- **触发条件**: `amsgrad=true` 且 `maxGradNormOptional` 为非连续 tensor（例如通过 transpose/slice 得到的视图）。
- **测试方案**: 构造非连续的 maxGradNormOptional tensor，设置 amsgrad=true，执行后检查 maxGradNormOptional 的值是否被正确更新。

---

## 汇总表

| # | 位置 | 类型 | 严重程度 | 简述 |
|---|------|------|----------|------|
| 1 | CheckShape L115-129 | 参数校验缺失 | 高 | grad shape 未与 varRef 校验一致性 |
| 2 | GetWorkspaceSize L218-219 | 参数校验缺失 | 高 | workspaceSize/executor 输出指针未判空 |
| 3 | GetWorkspaceSize L197-199 | 边界条件 | 低 | 标量tensor未做Contiguous转换 |
| 4 | CheckShape L124-127 | 错误处理 | 低 | 标量numel校验失败无错误日志 |
| 5 | GetWorkspaceSize L204-215 | 逻辑缺陷 | 高 | amsgrad=true时maxGradNormOptional未做ViewCopy回写 |
