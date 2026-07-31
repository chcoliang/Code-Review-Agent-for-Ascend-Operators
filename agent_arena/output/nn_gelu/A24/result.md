# GeLU op_api 层代码审查报告

审查文件: `aclnn_gelu.cpp`  
目标平台: Ascend 910B, CANN 8.5.0

---

## Bug #1: `aclnnGeluGetWorkspaceSize` 缺少对 `workspaceSize` 指针的空指针校验

| 属性 | 描述 |
|------|------|
| **位置** | 第 83-117 行，`aclnnGeluGetWorkspaceSize` 函数入口 |
| **类型** | 参数校验缺失 |
| **严重程度** | 高 (High) — 导致段错误/进程崩溃 |
| **描述** | 函数参数 `workspaceSize` 为 `uint64_t*` 类型，在第 97 行 (`*workspaceSize = 0`) 和第 115 行 (`*workspaceSize = uniqueExecutor->GetWorkspaceSize()`) 直接解引用，但函数入口未对其进行空指针检查。若调用者传入 `nullptr`，将触发空指针解引用导致 segfault。 |
| **触发输入** | `aclnnGeluGetWorkspaceSize(validSelf, validOut, nullptr, &executor)` |
| **预期异常** | 应返回 `ACLNN_ERR_PARAM_NULLPTR`，实际会触发 segfault (SIGSEGV) |

---

## Bug #2: `aclnnGeluGetWorkspaceSize` 缺少对 `executor` 指针的空指针校验

| 属性 | 描述 |
|------|------|
| **位置** | 第 83-117 行，`aclnnGeluGetWorkspaceSize` 函数入口 |
| **类型** | 参数校验缺失 |
| **严重程度** | 高 (High) — 导致段错误/进程崩溃 |
| **描述** | 函数参数 `executor` 为 `aclOpExecutor**` 类型，在第 98 行和第 116 行通过 `uniqueExecutor.ReleaseTo(executor)` 使用，但函数入口未对其进行空指针检查。若调用者传入 `nullptr`，`ReleaseTo` 内部解引用将导致未定义行为。 |
| **触发输入** | `aclnnGeluGetWorkspaceSize(validSelf, validOut, &workspaceSize, nullptr)` |
| **预期异常** | 应返回 `ACLNN_ERR_PARAM_NULLPTR`，实际会触发 segfault 或未定义行为 |

---

## Bug #3: `aclnnGelu` 缺少对 `executor` 和 `stream` 的空指针校验

| 属性 | 描述 |
|------|------|
| **位置** | 第 120-124 行，`aclnnGelu` 函数 |
| **类型** | 参数校验缺失 |
| **严重程度** | 高 (High) — 导致段错误/进程崩溃 |
| **描述** | `aclnnGelu` 作为执行阶段入口，未对 `executor` 和 `stream` 进行空指针校验，直接传递给 `CommonOpExecutorRun`。若用户未正确调用 GetWorkspaceSize 阶段或传入空 stream，将导致内部崩溃且无明确错误信息。 |
| **触发输入** | `aclnnGelu(workspace, workspaceSize, nullptr, stream)` 或 `aclnnGelu(workspace, workspaceSize, executor, nullptr)` |
| **预期异常** | 应返回 `ACLNN_ERR_PARAM_NULLPTR`，实际会触发 segfault 或不可预测行为 |

---

## Bug #4: 前向/反向算子对称性缺失 — 未校验 `approximate` 参数

| 属性 | 描述 |
|------|------|
| **位置** | 第 83 行，`aclnnGeluGetWorkspaceSize` 函数签名 |
| **类型** | 同族对称性 / 接口完整性 |
| **严重程度** | 中 (Medium) — 功能缺失，与 PyTorch 语义不对齐 |
| **描述** | PyTorch `torch.nn.functional.gelu` 支持 `approximate` 参数（`'none'` 或 `'tanh'`），对应反向 `gelu_backward` 也需要该参数来保证梯度正确。当前前向实现硬编码为 erf 近似（无 approximate 参数），若配套的 `aclnnGeluBackward` 实现支持 approximate 参数，则前向/反向语义不对称，反向传播时可能出现梯度计算错误。 |
| **触发输入** | 用户期望使用 `approximate='tanh'` 模式的 GeLU |
| **预期异常** | 应支持 approximate 参数或明确文档说明仅支持 erf 模式 |

---

## Bug #5: 空 tensor 路径下未校验 `out` 是否也为空

| 属性 | 描述 |
|------|------|
| **位置** | 第 96-99 行，空 tensor 处理分支 |
| **类型** | 边界条件 / 防御性校验不足 |
| **严重程度** | 低 (Low) — 逻辑不严谨，依赖上游 shape 检查 |
| **描述** | 当 `self->IsEmpty()` 为 true 时直接返回成功，未显式验证 `out` 也为空。虽然第 53 行的 shape 检查理论上保证了 self 和 out shape 一致，但如果 `OP_CHECK_SHAPE_NOT_EQUAL` 的实现仅比较 rank 而非每个维度（如 shape [0, 5] vs [5, 0] 都是 empty 但 shape 不同），可能存在 out 非空但仍进入该分支的边界场景。 |
| **触发输入** | 构造 self shape = [0]（empty），out shape = [0]（正常情况下 shape 检查通过），但若 shape 检查有宏实现缺陷则可能遗漏 |
| **预期异常** | 正常情况下不触发，属于防御性编程建议 |

---

## 审查总结

| 严重程度 | 数量 | 说明 |
|----------|------|------|
| 高 | 3 | 空指针解引用导致进程崩溃 |
| 中 | 1 | 接口对称性/功能完整性 |
| 低 | 1 | 边界条件防御性校验 |

核心问题集中在 **外部传入指针参数（workspaceSize、executor、stream）缺少空指针校验**，这是 op_api 层最基本的防御要求。在生产环境中，用户误传 nullptr 将导致整个推理进程崩溃而无法恢复。
