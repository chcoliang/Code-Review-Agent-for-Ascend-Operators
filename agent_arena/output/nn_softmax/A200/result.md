# Ascend NPU 算子代码审查报告

**文件**: `test_softmax_lifecycle.cpp`  
**审查重点**: 资源生命周期、内存泄漏

---

### Bug #1: aclTensorDesc 资源泄漏

| 项目 | 内容 |
|------|------|
| **位置** | 第 10 行，循环体内 |
| **类型** | 资源泄漏 (Resource Leak) |
| **严重程度** | 高 (High) |
| **描述** | 在 for 循环中通过 `aclCreateTensorDesc()` 创建了 `aclTensorDesc*` 对象，但循环体结束前未调用 `aclDestroyTensorDesc(desc)` 释放该资源。循环执行 100 次，导致 100 个 TensorDesc 对象泄漏。 |
| **触发条件** | 每次循环迭代均触发，程序运行即必现。累计泄漏 100 个 TensorDesc 描述符。 |
| **修复方案** | 在循环体末尾（第 12 行后）添加 `aclDestroyTensorDesc(desc);` |
| **测试方案** | 1. 使用 valgrind/ASAN 运行程序，检测未释放的内存块数量。<br>2. 在循环次数放大（如 100000 次）后监控进程 RSS 内存增长趋势。<br>3. 添加释放调用后重新运行，确认无泄漏报告。 |

---

### Bug #2: aclDataBuffer 资源泄漏

| 项目 | 内容 |
|------|------|
| **位置** | 第 11 行，循环体内 |
| **类型** | 资源泄漏 (Resource Leak) |
| **严重程度** | 高 (High) |
| **描述** | 在 for 循环中通过 `aclCreateDataBuffer()` 创建了 `aclDataBuffer*` 对象，但循环体结束前未调用 `aclDestroyDataBuffer(buf)` 释放该资源。循环执行 100 次，导致 100 个 DataBuffer 对象泄漏。 |
| **触发条件** | 每次循环迭代均触发，程序运行即必现。累计泄漏 100 个 DataBuffer 句柄。 |
| **修复方案** | 在循环体末尾（第 12 行后）添加 `aclDestroyDataBuffer(buf);` |
| **测试方案** | 1. 使用 valgrind/ASAN 运行程序，检测未释放的内存块数量。<br>2. 对比修复前后的内存占用。<br>3. 检查 ACL 内部资源计数器（如有）确认句柄全部归还。 |

---

### Bug #3: aclInit/aclrtSetDevice 返回值未检查

| 项目 | 内容 |
|------|------|
| **位置** | 第 5-6 行 |
| **类型** | 错误处理缺失 (Missing Error Handling) |
| **严重程度** | 中 (Medium) |
| **描述** | `aclInit()` 和 `aclrtSetDevice()` 的返回值未检查。若初始化或设备设置失败，后续操作将在无效上下文中执行，可能导致未定义行为或难以定位的崩溃。 |
| **触发条件** | 设备不可用、驱动未加载、权限不足等环境异常时触发。 |
| **修复方案** | 检查返回值，失败时打印错误信息并提前退出。 |
| **测试方案** | 在无 NPU 设备的环境中运行，验证程序能优雅退出而非崩溃。 |

---

### Bug #4: aclrtResetDevice/aclFinalize 返回值未检查

| 项目 | 内容 |
|------|------|
| **位置** | 第 15-16 行 |
| **类型** | 错误处理缺失 (Missing Error Handling) |
| **严重程度** | 低 (Low) |
| **描述** | `aclrtResetDevice()` 和 `aclFinalize()` 的返回值未检查。在存在资源泄漏的情况下，这些调用可能返回错误，但程序无法感知。 |
| **触发条件** | 当前面存在未释放资源时，Reset/Finalize 可能失败。 |
| **修复方案** | 检查返回值并记录日志。 |
| **测试方案** | 故意制造资源泄漏后调用 Reset，验证错误码是否被正确捕获和报告。 |

---

## 汇总表

| # | 位置 | Bug 类型 | 严重程度 | 简述 |
|---|------|----------|----------|------|
| 1 | L10 (循环内) | 资源泄漏 | 高 | `aclTensorDesc` 创建后未销毁，循环 100 次累计泄漏 |
| 2 | L11 (循环内) | 资源泄漏 | 高 | `aclDataBuffer` 创建后未销毁，循环 100 次累计泄漏 |
| 3 | L5-6 | 错误处理缺失 | 中 | `aclInit`/`aclrtSetDevice` 返回值未检查 |
| 4 | L15-16 | 错误处理缺失 | 低 | `aclrtResetDevice`/`aclFinalize` 返回值未检查 |

---

## 修复后参考代码

```cpp
#include <acl/acl.h>
#include <iostream>

int main() {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclInit failed, error: " << ret << std::endl;
        return -1;
    }
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed, error: " << ret << std::endl;
        aclFinalize();
        return -1;
    }

    for (int i = 0; i < 100; i++) {
        int64_t shape[] = {4, 1024};
        aclTensorDesc* desc = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
        aclDataBuffer* buf = aclCreateDataBuffer(nullptr, 4 * 1024 * 2);

        // ... 业务逻辑 ...

        aclDestroyDataBuffer(buf);
        aclDestroyTensorDesc(desc);
    }

    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
