# Code Review: test_leaky_relu_stress.cpp

## Bug 列表

### Bug 1: 循环内存分配未释放导致严重内存泄漏

- **位置**: 第 7-11 行，`for` 循环体内
- **类型**: 内存泄漏 (Memory Leak)
- **严重程度**: 严重 (Critical)
- **描述**: 循环 1000 次调用 `aclrtMalloc` 分配 16MB 设备内存，但从未调用 `aclrtFree(ws)` 释放。每次循环迭代中 `ws` 指针在下一轮被覆盖，导致前一次分配的内存句柄丢失，累计泄漏 16MB x 1000 = 16GB 设备内存。
- **触发条件**: 程序正常执行即可触发；在 NPU 设备内存有限的情况下（通常 8-64GB），极大概率在循环中途因 OOM 导致 `aclrtMalloc` 失败。
- **测试方案**: 
  1. 运行程序并监控 NPU 设备内存使用（通过 `npu-smi info`）；
  2. 观察内存持续增长直到 OOM；
  3. 修复后在循环末尾添加 `aclrtFree(ws)` 并验证内存稳定。

### Bug 2: aclrtMalloc 返回值未检查

- **位置**: 第 9 行
- **类型**: 资源管理 / 错误处理缺失
- **严重程度**: 高 (High)
- **描述**: `aclrtMalloc` 可能因设备内存不足而返回错误码（非 `ACL_SUCCESS`），但代码未检查返回值。在 Bug 1 导致内存耗尽后，后续分配必然失败，程序继续使用 `nullptr` 指针（未被更新），可能导致后续未定义行为。
- **触发条件**: 设备内存不足时 `aclrtMalloc` 返回失败，程序无感知继续执行。
- **测试方案**: 
  1. 在内存紧张环境下运行；
  2. 检查返回值是否为 `ACL_SUCCESS`，若失败应打印日志并退出或重试。

### Bug 3: aclInit / aclrtSetDevice / aclrtCreateStream 返回值未检查

- **位置**: 第 3、4、6 行
- **类型**: 资源管理 / 错误处理缺失
- **严重程度**: 中 (Medium)
- **描述**: `aclInit`、`aclrtSetDevice`、`aclrtCreateStream` 均可能失败（如设备不存在、驱动未加载），但返回值被忽略。若初始化失败，后续所有操作均在无效上下文中执行，行为未定义。
- **触发条件**: NPU 设备不可用、驱动未加载、或设备号不存在时触发。
- **测试方案**: 
  1. 在无 NPU 设备的环境中运行，观察是否有合理报错；
  2. 添加返回值检查和错误退出逻辑。

### Bug 4: 循环中 ws 变量作用域导致指针不可回收

- **位置**: 第 8-10 行
- **类型**: 资源管理设计缺陷
- **严重程度**: 严重 (Critical)
- **描述**: `ws` 在循环体内声明为局部变量，每次迭代结束后指针值丢失（栈帧局部变量超出作用域）。即使开发者后续想在循环外释放内存，也已无法获得之前分配的地址。这是一个设计性缺陷，使得内存泄漏无法通过简单的后置释放来修复，必须在每次迭代内释放或将指针保存到容器中。
- **触发条件**: 代码结构本身即存在此问题。
- **测试方案**: 
  1. 使用静态分析工具（如 cppcheck）检测；
  2. 修复方案：在循环体内添加 `aclrtFree(ws)` 或使用数组/vector 保存所有指针后统一释放。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简述 |
|------|------|----------|----------|------|
| 1 | 第 7-11 行 | 内存泄漏 | 严重 | 循环内 aclrtMalloc 16MB x 1000 次未释放，泄漏约 16GB |
| 2 | 第 9 行 | 错误处理缺失 | 高 | aclrtMalloc 返回值未检查，OOM 后无感知 |
| 3 | 第 3、4、6 行 | 错误处理缺失 | 中 | 初始化 API 返回值未检查 |
| 4 | 第 8-10 行 | 资源管理设计缺陷 | 严重 | ws 为循环局部变量，指针丢失无法事后回收 |

## 修复建议

```cpp
#include <acl/acl.h>
#include <cstdio>

int main() {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) { printf("aclInit failed: %d\n", ret); return -1; }

    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) { printf("aclrtSetDevice failed: %d\n", ret); aclFinalize(); return -1; }

    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) { printf("aclrtCreateStream failed: %d\n", ret); aclrtResetDevice(0); aclFinalize(); return -1; }

    for (int i = 0; i < 1000; i++) {
        void* ws = nullptr;
        ret = aclrtMalloc(&ws, 16*1024*1024, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            printf("aclrtMalloc failed at iter %d: %d\n", i, ret);
            break;
        }
        // ... 使用 ws 进行计算 ...
        aclrtFree(ws);  // 关键修复：每次迭代释放内存
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
