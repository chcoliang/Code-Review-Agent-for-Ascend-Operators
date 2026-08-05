# Ascend NPU 算子代码审查报告 — A186/test_mul_stress.cpp

## Bug 列表

### Bug 1: workspace 内存泄漏

- **位置**: 第 58-60 行（循环体内）
- **类型**: 内存泄漏（Device Memory Leak）
- **严重程度**: 严重（Critical）
- **描述**: 当 `workspaceSize > 0` 时，通过 `aclrtMalloc` 分配了 `workspaceAddr` 设备内存，但在循环迭代结束时从未调用 `aclrtFree(workspaceAddr)` 释放该内存。循环执行 1000 次，每次迭代都可能泄漏一块 workspace 内存，最终导致 NPU 设备内存耗尽。
- **触发条件**: `aclnnMulGetWorkspaceSize` 返回的 `workspaceSize > 0`（即算子需要额外工作空间），循环多次执行即触发累积泄漏。
- **测试方案**: 在循环中监控 `aclrtGetMemInfo` 报告的空闲设备内存，验证每次迭代后空闲内存持续减少；或将循环次数增大观察是否出现 `ACL_ERROR_RT_MEMORY_ALLOCATION` 错误。

---

### Bug 2: CreateAclTensor 失败时设备内存泄漏

- **位置**: 第 49-51 行（循环体内连续三次 `CreateAclTensor` 调用）
- **类型**: 资源管理缺陷 / 内存泄漏
- **严重程度**: 中等（Medium）
- **描述**: `CreateAclTensor` 内部先 `aclrtMalloc` 分配设备内存，再 `aclrtMemcpy`，若 `aclrtMemcpy` 失败则通过 `CHECK_RET` 直接返回错误码，但已分配的 `*deviceAddr` 不会被释放。此外，在 `main` 中如果第二或第三次 `CreateAclTensor` 调用失败，前面已经成功分配的设备内存和 tensor 对象也不会被清理（返回值未被检查）。
- **触发条件**: 设备内存不足导致中间某次 `aclrtMalloc` 或 `aclrtMemcpy` 失败时触发。
- **测试方案**: 通过预先占满大部分设备内存制造分配失败场景，观察程序退出后是否有未释放内存；使用 ASAN/设备内存追踪工具检测。

---

### Bug 3: CreateAclTensor 返回值未检查

- **位置**: 第 49-51 行
- **类型**: 资源管理缺陷 / 错误处理缺失
- **严重程度**: 中等（Medium）
- **描述**: `main` 函数中三次 `CreateAclTensor` 的返回值均未检查。若任一调用失败，后续代码将使用 `nullptr` 的 tensor 和无效的 deviceAddr 继续执行 `aclnnMulGetWorkspaceSize` 和 `aclnnMul`，可能导致未定义行为或段错误。
- **触发条件**: 设备内存耗尽或其他 ACL 运行时错误导致 `CreateAclTensor` 返回非零值。
- **测试方案**: 注入故障（如 mock `aclrtMalloc` 返回失败），验证程序是否正确处理错误路径。

---

### Bug 4: aclnnMulGetWorkspaceSize 返回值未检查

- **位置**: 第 55 行
- **类型**: 错误处理缺失
- **严重程度**: 低（Low）
- **描述**: `aclnnMulGetWorkspaceSize` 的返回值未检查。若该调用失败，`executor` 将处于未定义状态，后续 `aclnnMul` 可能崩溃或产生错误结果。
- **触发条件**: 传入无效 tensor 参数或内部资源分配失败时。
- **测试方案**: 传入 shape 不匹配的 tensor，验证返回值是否为错误码。

---

### Bug 5: aclrtMalloc (workspace) 返回值未检查

- **位置**: 第 59 行
- **类型**: 错误处理缺失
- **严重程度**: 低（Low）
- **描述**: workspace 的 `aclrtMalloc` 返回值未检查。若分配失败，`workspaceAddr` 仍为 `nullptr`，传入 `aclnnMul` 可能导致算子执行异常。
- **触发条件**: 设备内存耗尽时。
- **测试方案**: 制造内存压力场景，观察是否出现非预期的算子执行错误。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 58-60 行 | 内存泄漏 | 严重 | workspace 设备内存循环中未释放，1000 次迭代累积泄漏 |
| 2 | 第 49-51 行 / CreateAclTensor 内部 | 内存泄漏 | 中等 | CreateAclTensor 失败路径未释放已分配的设备内存 |
| 3 | 第 49-51 行 | 错误处理缺失 | 中等 | CreateAclTensor 返回值未检查，失败后继续执行导致未定义行为 |
| 4 | 第 55 行 | 错误处理缺失 | 低 | aclnnMulGetWorkspaceSize 返回值未检查 |
| 5 | 第 59 行 | 错误处理缺失 | 低 | workspace aclrtMalloc 返回值未检查 |
