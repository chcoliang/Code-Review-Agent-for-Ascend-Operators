# GeLU op_api 层代码审查报告

审查文件: `aclnn_gelu.cpp`  
目标平台: Ascend 910B, CANN 8.5.0

---

## Bug #1: 输出张量数据类型未校验

| 项目 | 内容 |
|------|------|
| **位置** | 第37-48行 `CheckDtypeValid` 函数 |
| **类型** | 参数校验缺失 |
| **严重程度** | 高 |
| **描述** | 函数仅校验了 `self` 的数据类型是否在支持列表中，对 `out` 的数据类型完全未做校验（第45行 `(void)out;` 明确忽略了输出参数）。如果 `out` 的 dtype 为不支持的类型（如 `DT_INT32`、`DT_INT8`、`DT_DOUBLE`），将导致后续 `ViewCopy` 时数据截断、精度丢失或内核执行异常。同时也缺少 `self` 与 `out` 之间的 dtype 一致性校验。 |
| **触发输入** | `self`: shape=[2,3], dtype=DT_FLOAT16; `out`: shape=[2,3], dtype=DT_INT32 |
| **预期异常** | 应在参数校验阶段返回 `ACLNN_ERR_PARAM_INVALID`，报告输出张量数据类型不支持或与输入不匹配 |

---

## Bug #2: `workspaceSize` 和 `executor` 指针未做空指针校验

| 项目 | 内容 |
|------|------|
| **位置** | 第83-84行 `aclnnGeluGetWorkspaceSize` 函数入口参数 |
| **类型** | 参数校验缺失 |
| **严重程度** | 高 |
| **描述** | 函数参数 `workspaceSize`（uint64_t*）和 `executor`（aclOpExecutor**）在使用前未做空指针检查。第97行 `*workspaceSize = 0` 和第115行 `*workspaceSize = uniqueExecutor->GetWorkspaceSize()` 以及第98/116行 `uniqueExecutor.ReleaseTo(executor)` 均会直接解引用这些指针。传入 nullptr 将导致段错误（SIGSEGV）。 |
| **触发输入** | `self`: 合法张量; `out`: 合法张量; `workspaceSize`: nullptr; `executor`: nullptr |
| **预期异常** | 应返回 `ACLNN_ERR_PARAM_NULLPTR`，而非进程崩溃 |

---

## Bug #3: GeLU 前向与反向（GeLUGrad）的同族对称性缺失 — 输出 dtype 推导不对称

| 项目 | 内容 |
|------|------|
| **位置** | 第37-48行 `CheckDtypeValid` 函数 |
| **类型** | 同族对称性 / 类型推导 |
| **严重程度** | 中 |
| **描述** | GeLU 作为激活算子，其前向实现未对 `out` 做任何类型推导或类型强制约束（即未强制 `out.dtype == self.dtype`）。标准实现中 GeLU 的输出类型应与输入严格一致。当用户创建一个 dtype 不同于 `self` 但仍在支持列表内的 `out` 张量时（如 self=FP32, out=FP16），不会报错，但会导致计算结果被静默截断/类型不匹配。 |
| **触发输入** | `self`: shape=[4,4], dtype=DT_FLOAT; `out`: shape=[4,4], dtype=DT_FLOAT16 |
| **预期异常** | 应返回 `ACLNN_ERR_PARAM_INVALID`，提示输入输出数据类型不一致 |

---

## Bug #4: 空 tensor 场景下未校验 `out` 的空状态一致性

| 项目 | 内容 |
|------|------|
| **位置** | 第95-99行 空tensor处理分支 |
| **类型** | 边界条件 |
| **严重程度** | 低 |
| **描述** | 当 `self->IsEmpty()` 为 true 时，直接返回成功，但未校验 `out` 是否也为空。由于前面 `CheckShape` 已验证 shape 一致，此处理论上 `out` 也应为空。但如果 `out` 是通过 stride 操控使得逻辑 shape 相同但实际存储非空的场景，可能存在不一致风险。此问题在当前 shape 检查严格匹配时风险较低。 |
| **触发输入** | `self`: shape=[0,3], dtype=DT_FLOAT; `out`: shape=[0,3], dtype=DT_FLOAT（但 out 有非零 storage offset） |
| **预期异常** | 行为应明确：要么同时判断 out 为空，要么对 out 进行显式清零 |

---

## Bug #5: `aclnnGelu` 执行函数缺少参数校验

| 项目 | 内容 |
|------|------|
| **位置** | 第120-124行 `aclnnGelu` 函数 |
| **类型** | 参数校验缺失 |
| **严重程度** | 中 |
| **描述** | 执行阶段函数对 `executor`、`stream` 和 `workspace`（当 workspaceSize > 0 时）均未做空指针校验，直接传递给 `CommonOpExecutorRun`。若调用方传入 nullptr 的 executor 或 stream，可能导致空指针解引用崩溃。虽然这可能依赖 `CommonOpExecutorRun` 内部校验，但作为对外 API 入口应自行防护。 |
| **触发输入** | `workspace`: nullptr; `workspaceSize`: 0; `executor`: nullptr; `stream`: 合法stream |
| **预期异常** | 应返回 `ACLNN_ERR_PARAM_NULLPTR`，而非进程崩溃 |

---

## 审查总结

| 严重程度 | 数量 | 说明 |
|---------|------|------|
| 高 | 2 | 输出dtype未校验、关键指针未校验 |
| 中 | 2 | 输入输出dtype一致性、执行函数参数校验 |
| 低 | 1 | 空tensor边界条件 |

核心风险：输出张量 `out` 的数据类型完全未被校验，是最关键的遗漏，可能导致静默数据损坏或内核崩溃。
