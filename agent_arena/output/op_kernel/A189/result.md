# Ascend NPU 算子代码审查报告 — A189 test_mul_lifecycle.cpp

## Bug 列表

### Bug 1: aclTensor 对象未销毁导致内存泄漏

| 项目 | 内容 |
|------|------|
| **位置** | 第 65-67 行（被注释掉的 `aclDestroyTensor` 调用） |
| **类型** | 资源泄漏 (Resource Leak) |
| **严重程度** | 严重 (Critical) |
| **描述** | 每次循环迭代中通过 `aclCreateTensor` 创建了 3 个 `aclTensor` 对象（`self`、`other`、`out`），但对应的 `aclDestroyTensor` 调用被注释掉。循环执行 10000 次，将累计泄漏 30000 个 `aclTensor` 对象的元数据内存（包括 shape、strides 等内部分配），最终导致宿主侧内存耗尽或 ACL 内部资源池枯竭。 |
| **触发条件** | 程序正常执行即可触发；循环迭代次数越多，泄漏量线性增长。 |
| **修复方案** | 取消第 65-67 行注释，在 `aclrtFree` 之前调用 `aclDestroyTensor(self)`、`aclDestroyTensor(other)`、`aclDestroyTensor(out)` 销毁 tensor 描述符。 |
| **测试方案** | 1. 使用 Valgrind/AddressSanitizer 运行程序，检查是否报告 "definitely lost" 内存。<br>2. 监控进程 RSS 随迭代增长情况，修复后应保持平稳。<br>3. 减少迭代次数运行对比修复前后的 ACL 内部资源计数。 |

---

### Bug 2: CreateAclTensor 中 aclrtMemcpy 失败时设备内存泄漏

| 项目 | 内容 |
|------|------|
| **位置** | 第 26-27 行 `CHECK_RET(ret == ACL_SUCCESS, return ret);` |
| **类型** | 错误路径资源泄漏 (Error-path Leak) |
| **严重程度** | 中等 (Medium) |
| **描述** | 当 `aclrtMalloc` 成功（第 24 行）但随后 `aclrtMemcpy` 失败时（第 26 行），函数直接 `return ret`，已分配的 `*deviceAddr` 设备内存未被释放，造成设备内存泄漏。 |
| **触发条件** | `aclrtMemcpy` 因 DMA 错误、参数异常或设备繁忙返回非 `ACL_SUCCESS` 时触发。 |
| **修复方案** | 在 `aclrtMemcpy` 失败的 CHECK_RET 分支中，先调用 `aclrtFree(*deviceAddr)` 并将 `*deviceAddr` 置 nullptr，再返回错误码。 |
| **测试方案** | 1. Mock `aclrtMemcpy` 返回失败，验证 `*deviceAddr` 是否被正确释放。<br>2. 故障注入测试：构造非法 size 使 memcpy 失败，用工具检测设备内存泄漏。 |

---

### Bug 3: CreateAclTensor 失败时主循环未处理错误，继续使用空指针

| 项目 | 内容 |
|------|------|
| **位置** | 第 49-51 行（主循环中调用 `CreateAclTensor` 未检查返回值） |
| **类型** | 错误处理缺失 / 空指针解引用风险 |
| **严重程度** | 中等 (Medium) |
| **描述** | `CreateAclTensor` 可能返回非零错误码（malloc 或 memcpy 失败），但调用处未检查返回值。若失败，`self`/`other`/`out` 仍为 nullptr，后续传入 `aclnnMulGetWorkspaceSize` 将导致未定义行为或程序崩溃。同时前序成功分配的设备内存也不会被正确清理。 |
| **触发条件** | 设备内存不足（循环 10000 次逐步耗尽设备内存时尤为可能）。 |
| **修复方案** | 检查每次 `CreateAclTensor` 返回值，失败时释放已分配资源并 `break` 或 `continue`。 |
| **测试方案** | 1. 限制设备内存配额，验证程序是否能优雅退出而非崩溃。<br>2. 在循环中间注入 malloc 失败，确认无 segfault。 |

---

### Bug 4: workspace 内存在 workspaceSize == 0 且 aclnnMul 异常时潜在问题

| 项目 | 内容 |
|------|------|
| **位置** | 第 57-61 行 |
| **类型** | 逻辑缺陷 (Minor) |
| **严重程度** | 低 (Low) |
| **描述** | `aclnnMulGetWorkspaceSize` 的返回值未检查。若该调用失败，`workspaceSize` 和 `executor` 的值不确定，后续 `aclnnMul` 使用无效 executor 会导致未定义行为。 |
| **触发条件** | 传入的 tensor 参数不合法或内部错误导致 GetWorkspaceSize 失败。 |
| **修复方案** | 检查 `aclnnMulGetWorkspaceSize` 返回值，失败时跳过执行并清理资源。 |
| **测试方案** | 传入 shape 不匹配的 tensor，验证错误是否被正确捕获。 |

---

## 汇总表

| # | 位置 | Bug 类型 | 严重程度 | 根因 |
|---|------|----------|----------|------|
| 1 | L65-67 | aclTensor 资源泄漏 | 严重 | `aclDestroyTensor` 被注释，循环中创建的 tensor 描述符从未释放 |
| 2 | L26-27 | 错误路径设备内存泄漏 | 中等 | memcpy 失败时未释放已 malloc 的设备内存 |
| 3 | L49-51 | 错误处理缺失 / 空指针风险 | 中等 | CreateAclTensor 返回值未检查，失败后继续使用空指针 |
| 4 | L55 | 返回值未检查 | 低 | aclnnMulGetWorkspaceSize 失败后使用无效 executor |

## 核心修复建议

```cpp
// 循环体内关键修复（Bug 1）：取消注释 tensor 销毁
aclDestroyTensor(self);
aclDestroyTensor(other);
aclDestroyTensor(out);

// Bug 2 修复：CreateAclTensor 中 memcpy 失败路径
ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
if (ret != ACL_SUCCESS) {
    aclrtFree(*deviceAddr);
    *deviceAddr = nullptr;
    return ret;
}
```
