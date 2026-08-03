# Code Review Report: test_gelu_stress.cpp

## 审查文件
`agent_arena/cases/nn_gelu/A205/test_gelu_stress.cpp`

---

### Bug 1: 循环内存分配未释放导致严重内存泄漏

| 项目 | 详情 |
|------|------|
| **位置** | 第 7-11 行，`for` 循环体内部 |
| **类型** | 内存泄漏 (Memory Leak) |
| **严重程度** | 致命 (Critical) |
| **描述** | 循环 1000 次调用 `aclrtMalloc` 分配 16MB 设备内存，但循环体内从未调用 `aclrtFree(ws)` 释放内存。每次迭代局部变量 `ws` 被重新声明并覆盖，前一次分配的地址丢失，导致累计泄漏 1000 × 16MB = 16GB 设备内存。 |
| **触发条件** | 程序正常执行即触发。循环运行期间 NPU 设备内存持续增长，极大概率导致 OOM（Out of Memory）错误，使后续 `aclrtMalloc` 调用失败。 |
| **修复建议** | 在循环末尾添加 `aclrtFree(ws);` |
| **测试方案** | 1. 使用 `aclrtGetMemInfo` 在循环前后监测设备可用内存，验证是否持续减少。<br>2. 检查 `aclrtMalloc` 返回值，观察循环后期是否返回内存不足错误码。<br>3. 使用 Ascend 内存分析工具 (msprof) 检测内存泄漏。 |

---

### Bug 2: aclrtMalloc 返回值未检查

| 项目 | 详情 |
|------|------|
| **位置** | 第 9 行 |
| **类型** | 资源管理缺陷 (Missing Error Handling) |
| **严重程度** | 高 (High) |
| **描述** | `aclrtMalloc` 的返回值（`aclError`）未被检查。当设备内存耗尽时，该调用将失败，`ws` 保持 `nullptr`，但程序无任何感知，若后续对 `ws` 进行操作将导致未定义行为。 |
| **触发条件** | 当设备内存不足（由 Bug 1 的泄漏加速引发），`aclrtMalloc` 返回非零错误码但被忽略。 |
| **修复建议** | 检查返回值：`if (aclrtMalloc(&ws, ...) != ACL_SUCCESS) { /* 错误处理 */ }` |
| **测试方案** | 在内存紧张环境下运行，检查程序是否能正确感知并处理分配失败。 |

---

### Bug 3: aclInit / aclrtSetDevice / aclrtCreateStream 返回值未检查

| 项目 | 详情 |
|------|------|
| **位置** | 第 3、4、6 行 |
| **类型** | 资源管理缺陷 (Missing Error Handling) |
| **严重程度** | 中 (Medium) |
| **描述** | `aclInit`、`aclrtSetDevice`、`aclrtCreateStream` 的返回值均未检查。若初始化或设备设置失败，程序继续运行将导致后续所有 ACL 调用行为未定义。 |
| **触发条件** | 设备不可用、驱动未加载或资源不足时触发。 |
| **修复建议** | 逐一检查返回值，失败时打印错误并提前退出。 |
| **测试方案** | 在无 NPU 设备环境下运行，验证程序是否优雅退出而非崩溃。 |

---

## 汇总表

| # | 位置 | Bug 类型 | 严重程度 | 核心问题 |
|---|------|----------|----------|----------|
| 1 | 第 7-11 行 | 内存泄漏 | 致命 | 循环内 `aclrtMalloc` 分配 16MB×1000 次从未释放，累计泄漏 16GB 设备内存 |
| 2 | 第 9 行 | 返回值未检查 | 高 | `aclrtMalloc` 失败时无感知，可能操作空指针 |
| 3 | 第 3/4/6 行 | 返回值未检查 | 中 | 初始化类 API 失败后程序盲目继续执行 |

---

## 修复后参考代码

```cpp
#include <acl/acl.h>
#include <cstdio>

int main() {
    if (aclInit(nullptr) != ACL_SUCCESS) { return -1; }
    if (aclrtSetDevice(0) != ACL_SUCCESS) { aclFinalize(); return -1; }

    aclrtStream stream;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        aclrtResetDevice(0);
        aclFinalize();
        return -1;
    }

    for (int i = 0; i < 1000; i++) {
        void* ws = nullptr;
        aclError ret = aclrtMalloc(&ws, 16 * 1024 * 1024, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "aclrtMalloc failed at iter %d, error: %d\n", i, ret);
            break;
        }
        // ... 使用 ws 进行计算 ...
        aclrtFree(ws);  // 关键修复：释放内存
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
