# Code Review: aclnn_apply_adam_w.cpp (A157)

## Bug 列表

### Bug 1: amsgrad 模式下 maxGradNormOptional 非连续时结果丢失

- **位置**: 第 189-191 行 及 第 205-216 行
- **类型**: 逻辑错误 / 数据丢失
- **严重程度**: 高
- **描述**: 当 `amsgrad=true` 且 `maxGradNormOptional` 指向非连续 tensor 时，代码在第 189 行将其转为连续副本 `maxGradNormContiguous`，并传入 `ApplyAdamW` 进行计算（AdamW amsgrad 模式会就地更新 max gradient norm）。然而在第 205-216 行的 ViewCopy 回写逻辑中，只对 `varRef`、`mRef`、`vRef` 进行了非连续场景的回写，遗漏了 `maxGradNormOptional` 的回写。这导致当 `maxGradNormOptional` 非连续时，其更新结果被丢弃。
- **触发条件**: 调用方传入的 `maxGradNormOptional` tensor 存储格式非连续（如带 stride 的视图），且 `amsgrad=true`。
- **测试方案**: 构造一个非连续的 maxGradNorm tensor（如通过 slice 获得的 view），设置 `amsgrad=true`，执行 AdamW 后检查 maxGradNorm 是否正确更新。

---

### Bug 2: 输出指针参数 workspaceSize 和 executor 未做空指针校验

- **位置**: 第 156 行（函数签名）、第 171/219/220 行（解引用处）
- **类型**: 参数校验缺失 / 潜在段错误
- **严重程度**: 中
- **描述**: `aclnnApplyAdamWGetWorkspaceSize` 函数接收 `uint64_t* workspaceSize` 和 `aclOpExecutor** executor` 两个输出参数，但在使用前未做空指针检查。若调用方误传 `nullptr`，将在第 171 行（`*workspaceSize = 0`）或第 219 行（`*workspaceSize = ...`）及第 172/220 行（`ReleaseTo(executor)`）触发未定义行为（段错误）。
- **触发条件**: 外部调用方传入 `workspaceSize=nullptr` 或 `executor=nullptr`。
- **测试方案**: 分别传入 `nullptr` 作为 `workspaceSize` 和 `executor`，验证是否安全返回错误码而非崩溃。

---

### Bug 3: 标量 tensor（beta1Power 等）未转为连续格式

- **位置**: 第 198-200 行
- **类型**: 潜在计算错误
- **严重程度**: 低
- **描述**: `beta1Power`、`beta2Power`、`lr`、`weightDecay`、`beta1`、`beta2`、`eps` 这些标量 tensor 在传入 `l0op::ApplyAdamW` 之前未调用 `l0op::Contiguous` 进行连续化处理。虽然它们是 numel=1 的标量，大多数情况下本身就是连续的，但如果以非连续格式（如带 offset 的 view）传入，底层 kernel 可能读取到错误数据。
- **触发条件**: 标量 tensor 以 storage offset 或非标准 stride 的视图形式传入。
- **测试方案**: 构造带有 storage_offset 的标量 tensor 作为 lr/beta1 等参数传入，对比结果与连续标量输入的结果是否一致。

---

### Bug 4: ASCEND910B 支持 DT_INT32 类型不合理

- **位置**: 第 35-36 行
- **类型**: 精度/语义错误
- **严重程度**: 中
- **描述**: AdamW 优化器涉及浮点除法、指数衰减、平方根等运算，对整数类型 `DT_INT32` 不具有数学意义。在 `ASCEND910B_DTYPE_SUPPORT_LIST` 中包含 `DT_INT32` 会导致：(1) 整数溢出；(2) 除法截断使动量/方差无法正确累积；(3) eps 加法在整数下失效。若用户误传 INT32 tensor，通过了类型校验后将产生完全错误的计算结果而无任何报错。
- **触发条件**: 在 ASCEND910B/ASCEND910_93 平台上传入 dtype 为 INT32 的参数 tensor。
- **测试方案**: 传入 INT32 类型的 var/m/v/grad tensor，执行 AdamW 并验证结果是否有数学意义（应返回错误而非静默计算）。

---

### Bug 5: CheckShape 中标量 tensor Numel 校验失败时缺少错误日志

- **位置**: 第 125-128 行
- **类型**: 可维护性 / 错误诊断
- **严重程度**: 低
- **描述**: 当 `beta1Power`、`beta2Power`、`lr`、`weightDecay`、`beta1`、`beta2`、`eps` 中任一 Numel 不为 1 时，函数直接 `return false`，没有打印任何错误日志说明是哪个参数的 shape 不满足标量约束。上层 `CheckParams` 仅返回 `ACLNN_ERR_PARAM_INVALID`，用户无法定位问题。相比之下，`OP_CHECK_SHAPE_NOT_EQUAL` 宏自带错误日志。
- **触发条件**: 任何标量参数的 numel != 1。
- **测试方案**: 传入 shape 为 [2] 的 beta1 tensor，检查是否有明确的错误日志指出哪个参数出错。

---

## 汇总表

| # | 位置 | 类型 | 严重程度 | 简述 |
|---|------|------|----------|------|
| 1 | L189-216 | 逻辑错误/数据丢失 | 高 | amsgrad 模式下 maxGradNormOptional 非连续时更新结果未回写 |
| 2 | L156,171,219,220 | 参数校验缺失 | 中 | workspaceSize/executor 输出指针未校验空指针 |
| 3 | L198-200 | 潜在计算错误 | 低 | 标量 tensor 未做 Contiguous 处理 |
| 4 | L35-36 | 精度/语义错误 | 中 | ASCEND910B 支持列表含 DT_INT32，AdamW 对整数无意义 |
| 5 | L125-128 | 可维护性 | 低 | 标量 shape 校验失败无错误日志，难以定位问题 |
