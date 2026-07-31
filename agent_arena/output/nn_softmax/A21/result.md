# Softmax op_api 层代码审查报告

**审查文件**: `aclnn_softmax.cpp`  
**目标平台**: Ascend 910B, CANN 8.5.0

---

## Bug #1: workspaceSize 和 executor 指针未做空指针校验

| 属性 | 内容 |
|------|------|
| **位置** | 第108-109行 (`aclnnSoftmaxGetWorkspaceSize` 函数签名), 第138-139行 (解引用处) |
| **类型** | 参数校验缺失 |
| **严重程度** | 高 |
| **描述** | 函数参数 `uint64_t* workspaceSize` 和 `aclOpExecutor** executor` 在第138行 `*workspaceSize = ...` 和第139行 `uniqueExecutor.ReleaseTo(executor)` 处直接解引用，但函数入口未对这两个指针做空指针检查。若调用者传入 `nullptr`，将导致段错误(SIGSEGV)。 |
| **触发输入** | `aclnnSoftmaxGetWorkspaceSize(validSelf, 0, validOut, nullptr, &executor)` 或 `aclnnSoftmaxGetWorkspaceSize(validSelf, 0, validOut, &ws, nullptr)` |
| **预期异常** | 应返回 `ACLNN_ERR_PARAM_NULLPTR` 错误码，而非进程崩溃 |

---

## Bug #2: 空tensor提前返回跳过shape/dtype/dim校验

| 属性 | 内容 |
|------|------|
| **位置** | 第93-95行 (`CheckParams` 函数) |
| **类型** | 参数校验不完整 / 边界条件 |
| **严重程度** | 中 |
| **描述** | 当 `self->IsEmpty()` 为 true 时，`CheckParams` 直接返回 `ACLNN_SUCCESS`，跳过了dtype校验、dim范围校验和shape一致性校验。这意味着: (1) `out` 的shape可以与 `self` 不一致; (2) `dim` 可以是任意非法值; (3) `out` 的dtype可以不在支持列表中。更严重的是，返回成功后 `aclnnSoftmaxGetWorkspaceSize` 继续执行第122行的 `Contiguous`、第126行的 `SoftmaxV2` 等操作，可能因非法dim导致底层算子异常。 |
| **触发输入** | `self` 为 shape=[0,3] 的空tensor，`dim=100`（非法值），`out` 为 shape=[5,5] 的非空tensor |
| **预期异常** | 应返回 `ACLNN_ERR_PARAM_INVALID`，实际返回 `ACLNN_SUCCESS` 并继续执行后续逻辑 |

---

## Bug #3: 缺少 self 与 out 之间的数据类型一致性/兼容性校验

| 属性 | 内容 |
|------|------|
| **位置** | 第58-64行 (`CheckDtypeValid` 函数) |
| **类型** | 参数校验缺失 |
| **严重程度** | 中 |
| **描述** | `CheckDtypeValid` 仅分别校验 `self` 和 `out` 的dtype是否在支持列表中，但未校验两者之间的类型关系。Softmax 语义要求输出类型应与输入一致（或为对应的浮点提升类型）。当 `self` 为 FP16 而 `out` 为 BF16 时，两者都在支持列表中，校验通过，但第130行的 `Cast` 操作虽能执行，语义上 FP16→BF16 的 softmax 输出并非 PyTorch 标准行为，可能导致精度问题或与框架层期望不符。 |
| **触发输入** | `self` dtype=DT_FLOAT16, `out` dtype=DT_BF16, shape一致，dim=0 |
| **预期异常** | 应返回 `ACLNN_ERR_PARAM_INVALID` 表示dtype不匹配 |

---

## Bug #4: CheckShape 未对 out 进行维度上限校验

| 属性 | 内容 |
|------|------|
| **位置** | 第80-85行 (`CheckShape` 函数) |
| **类型** | 参数校验不对称 |
| **严重程度** | 低 |
| **描述** | `CheckShape` 中仅对 `self` 做了 `OP_CHECK_MAX_DIM(self, AXIS_LIMIT, ...)` 检查，未对 `out` 做同样校验。虽然后续有 shape 相等检查，但在 `self` 维度为0（标量）的场景下，`OP_CHECK_MAX_DIM` 可能通过而 `out` 可能具有不同维度数。逻辑上应保持同族对称性，对输入输出均做维度上限检查。 |
| **触发输入** | 需要找到 `OP_CHECK_SHAPE_NOT_EQUAL` 无法捕获的边界 case（如标量 self 与高维 out 的比较实现依赖） |
| **预期异常** | 应对 out 超过8维也报错 |

---

## Bug #5: CheckNotNull 中 out 参数缺少 const 限定符（同族对称性问题）

| 属性 | 内容 |
|------|------|
| **位置** | 第32行 (`CheckNotNull` 函数签名) |
| **类型** | 接口设计 / 同族对称性 |
| **严重程度** | 低 |
| **描述** | `CheckNotNull(const aclTensor *self, aclTensor *out)` 中 `self` 有 `const` 限定但 `out` 没有。而 `CheckDtypeValid(const aclTensor *self, const aclTensor *out)` 中 `out` 有 `const`。空指针检查不需要修改对象，应统一使用 `const`。这不会导致运行时错误，但破坏了接口一致性，且在某些调用场景下可能阻止传入 const 指针。 |
| **触发输入** | 编译期问题，非运行时触发 |
| **预期异常** | 若上层传入 `const aclTensor*` 类型的 out，将编译报错 |

---

## Bug #6: 注释错误 — "SoftmaxGrad" 应为 "SoftmaxV2"

| 属性 | 内容 |
|------|------|
| **位置** | 第125行 |
| **类型** | 注释错误 (copy-paste) |
| **严重程度** | 极低（代码可维护性） |
| **描述** | 注释写 `// 调用SoftmaxGrad算子kernel`，但实际调用的是 `l0op::SoftmaxV2`。这是 softmax forward 实现，不是 backward/grad。疑似从 softmax_backward 实现复制而来，未更新注释。 |
| **触发输入** | N/A |
| **预期异常** | N/A（不影响运行时行为） |

---

## 总结

| 严重程度 | 数量 | 主要风险 |
|---------|------|---------|
| 高 | 1 | 空指针解引用导致进程崩溃 |
| 中 | 2 | 校验遗漏导致非法参数透传到底层算子 |
| 低 | 2 | 接口对称性/健壮性 |
| 极低 | 1 | 注释误导 |
