# Code Review Report: test_matmul_lifecycle.cpp

## Bug List

### Bug 1: aclTensorDesc 循环内创建未释放（资源泄漏）

| 项目 | 内容 |
|------|------|
| **位置** | 第 7 行，`for` 循环体内（第 5-10 行） |
| **类型** | 资源泄漏（Memory/Resource Leak） |
| **严重程度** | 严重（Critical） |
| **描述** | 每次循环调用 `aclCreateTensorDesc()` 创建 TensorDesc 对象并赋值给局部指针 `d`，但循环体内及循环后均未调用 `aclDestroyTensorDesc(d)` 释放该资源。循环 100 次将泄漏 100 个 TensorDesc 对象。 |
| **触发条件** | 程序正常执行即触发，循环每迭代一次泄漏一个 TensorDesc。 |
| **修复方案** | 在循环体末尾（第 9 行）添加 `aclDestroyTensorDesc(d);` |
| **测试方案** | 使用 ASAN/Valgrind 运行程序，验证无内存泄漏；或在循环次数增大后监控进程 RSS 内存增长。 |

---

### Bug 2: aclDataBuffer 循环内创建未释放（资源泄漏）

| 项目 | 内容 |
|------|------|
| **位置** | 第 8 行，`for` 循环体内（第 5-10 行） |
| **类型** | 资源泄漏（Memory/Resource Leak） |
| **严重程度** | 严重（Critical） |
| **描述** | 每次循环调用 `aclCreateDataBuffer()` 创建 DataBuffer 对象并赋值给局部指针 `b`，但循环体内及循环后均未调用 `aclDestroyDataBuffer(b)` 释放该资源。循环 100 次将泄漏 100 个 DataBuffer 对象。 |
| **触发条件** | 程序正常执行即触发，循环每迭代一次泄漏一个 DataBuffer。 |
| **修复方案** | 在循环体末尾（第 9 行）添加 `aclDestroyDataBuffer(b);` |
| **测试方案** | 使用 ASAN/Valgrind 运行程序，验证无内存泄漏；监控 ACL 内部资源计数。 |

---

### Bug 3: aclInit/aclrtSetDevice 返回值未检查（健壮性缺陷）

| 项目 | 内容 |
|------|------|
| **位置** | 第 3-4 行 |
| **类型** | 错误处理缺失 |
| **严重程度** | 中等（Medium） |
| **描述** | `aclInit()` 和 `aclrtSetDevice()` 的返回值（`aclError`）未检查。若初始化或设备设置失败，程序继续执行后续 ACL 调用将产生未定义行为。 |
| **触发条件** | 设备不可用、驱动未加载、重复初始化等异常场景。 |
| **修复方案** | 检查返回值，失败时打印错误并提前退出。 |
| **测试方案** | 在无 NPU 设备环境运行，验证程序能正确报错退出而非崩溃。 |

---

### Bug 4: aclCreateDataBuffer 传入 nullptr 作为设备内存地址

| 项目 | 内容 |
|------|------|
| **位置** | 第 8 行 |
| **类型** | 逻辑错误 / 潜在空指针问题 |
| **严重程度** | 中等（Medium） |
| **描述** | `aclCreateDataBuffer(nullptr, 2048*2048*2)` 创建了一个大小为 8MB 但设备内存指针为空的 DataBuffer。正常流程应先通过 `aclrtMalloc` 分配设备内存再传入。若该 buffer 后续用于算子执行，将导致空指针访问或硬件异常。 |
| **触发条件** | 将该 DataBuffer 传入 `aclopExecuteV2` 等算子执行接口时触发。 |
| **修复方案** | 先调用 `aclrtMalloc(&devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST)` 分配设备内存，再将 `devPtr` 传入 `aclCreateDataBuffer`。 |
| **测试方案** | 将 DataBuffer 用于实际 MatMul 算子执行，验证是否正确计算而非崩溃。 |

---

## 汇总表

| # | 位置 | Bug 类型 | 严重程度 | 简述 |
|---|------|----------|----------|------|
| 1 | L7 (循环内) | 资源泄漏 | Critical | TensorDesc 创建后未调用 aclDestroyTensorDesc 释放 |
| 2 | L8 (循环内) | 资源泄漏 | Critical | DataBuffer 创建后未调用 aclDestroyDataBuffer 释放 |
| 3 | L3-4 | 错误处理缺失 | Medium | aclInit/aclrtSetDevice 返回值未检查 |
| 4 | L8 | 逻辑错误 | Medium | DataBuffer 使用 nullptr 作为设备内存地址 |

## 修复后参考代码

```cpp
#include <acl/acl.h>
#include <cstdio>

int main() {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) { fprintf(stderr, "aclInit failed: %d\n", ret); return 1; }
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) { fprintf(stderr, "aclrtSetDevice failed: %d\n", ret); aclFinalize(); return 1; }

    for (int i = 0; i < 100; i++) {
        int64_t s[] = {2048, 2048};
        aclTensorDesc* d = aclCreateTensorDesc(ACL_FLOAT16, 2, s, ACL_FORMAT_ND);
        void* devPtr = nullptr;
        aclrtMalloc(&devPtr, 2048*2048*2, ACL_MEM_MALLOC_HUGE_FIRST);
        aclDataBuffer* b = aclCreateDataBuffer(devPtr, 2048*2048*2);

        // ... use d and b ...

        aclDestroyDataBuffer(b);
        aclrtFree(devPtr);
        aclDestroyTensorDesc(d);
    }

    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
