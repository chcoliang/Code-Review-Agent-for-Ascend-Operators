# Ascend NPU 算子代码审查报告

**文件**: `test_scaled_softmax_lifecycle.cpp`  
**审查重点**: 资源生命周期、内存泄漏

---

### Bug 1: aclTensorDesc 资源泄漏

- **位置**: 第 7 行，循环体内（第 5-10 行）
- **类型**: 资源泄漏（Resource Leak）
- **严重程度**: 严重（Critical）
- **描述**: 在 for 循环中每次迭代通过 `aclCreateTensorDesc()` 创建了 `aclTensorDesc*` 对象，但循环体内没有调用 `aclDestroyTensorDesc(d)` 释放该资源。循环执行 100 次，导致 100 个 TensorDesc 对象泄漏。
- **触发条件**: 每次循环迭代均触发，程序运行即 100% 复现。累计泄漏随循环次数线性增长。
- **修复方案**: 在循环体末尾（第 9 行位置）添加 `aclDestroyTensorDesc(d);`
- **测试方案**: 使用 Valgrind/ASan 或 ACL 内置的资源追踪工具运行程序，检查是否报告 TensorDesc 相关的内存泄漏；对比修复前后的内存占用曲线。

---

### Bug 2: aclDataBuffer 资源泄漏

- **位置**: 第 8 行，循环体内（第 5-10 行）
- **类型**: 资源泄漏（Resource Leak）
- **严重程度**: 严重（Critical）
- **描述**: 在 for 循环中每次迭代通过 `aclCreateDataBuffer()` 创建了 `aclDataBuffer*` 对象，但循环体内没有调用 `aclDestroyDataBuffer(b)` 释放该资源。循环执行 100 次，导致 100 个 DataBuffer 描述符对象泄漏。
- **触发条件**: 每次循环迭代均触发，程序运行即 100% 复现。
- **修复方案**: 在循环体末尾添加 `aclDestroyDataBuffer(b);`
- **测试方案**: 同 Bug 1，利用内存检测工具验证 DataBuffer 资源是否正确回收。

---

### Bug 3: aclCreateDataBuffer 传入 nullptr 但声明了非零大小

- **位置**: 第 8 行
- **类型**: 逻辑缺陷 / 潜在未定义行为
- **严重程度**: 中等（Medium）
- **描述**: `aclCreateDataBuffer(nullptr, 4*8*512*512*2)` 创建了一个声称拥有 16 MB 数据的 buffer 描述符，但底层设备内存指针为 `nullptr`。若后续算子执行时引用此 buffer，将导致空指针访问。即使当前代码未执行算子，这也违反了资源配对使用的最佳实践——正确流程应先通过 `aclrtMalloc` 分配设备内存，再传入有效指针。
- **触发条件**: 当该 DataBuffer 被传递给算子执行接口时触发非法内存访问。
- **修复方案**: 在创建 DataBuffer 前通过 `aclrtMalloc(&devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST)` 分配设备内存，并在释放 DataBuffer 后调用 `aclrtFree(devPtr)`。
- **测试方案**: 补全算子调用流程，验证传入 nullptr 的 DataBuffer 是否导致运行时错误。

---

### Bug 4: aclrtSetDevice 缺少 Stream/Context 管理

- **位置**: 第 4 行与第 11 行之间
- **类型**: 资源生命周期不完整（Minor）
- **严重程度**: 低（Low）
- **描述**: 调用了 `aclrtSetDevice(0)` 会隐式创建默认 context，但未显式管理 context 生命周期。虽然 `aclrtResetDevice` 会清理隐式 context，但在生产代码中缺乏显式 context 管理可能导致多线程场景下的资源竞争。
- **触发条件**: 多线程或多 stream 并发场景。
- **修复方案**: 显式创建和销毁 context：`aclrtCreateContext` / `aclrtDestroyContext`。
- **测试方案**: 在多线程环境下运行，检查是否存在 context 资源竞争。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 缺失的释放调用 |
|------|------|------|----------|----------------|
| 1 | 第 7 行（循环内） | 资源泄漏 | Critical | `aclDestroyTensorDesc(d)` |
| 2 | 第 8 行（循环内） | 资源泄漏 | Critical | `aclDestroyDataBuffer(b)` |
| 3 | 第 8 行 | 逻辑缺陷 | Medium | 需先 `aclrtMalloc` 再传入有效指针 |
| 4 | 第 4 行 | 生命周期不完整 | Low | 建议显式 context 管理 |

---

## 修复后参考代码

```cpp
#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    for (int i = 0; i < 100; i++) {
        int64_t s[] = {4, 8, 512, 512};
        aclTensorDesc* d = aclCreateTensorDesc(ACL_FLOAT16, 4, s, ACL_FORMAT_ND);
        void* devPtr = nullptr;
        size_t size = 4 * 8 * 512 * 512 * 2;
        aclrtMalloc(&devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclDataBuffer* b = aclCreateDataBuffer(devPtr, size);

        // ... 算子执行 ...

        aclDestroyDataBuffer(b);
        aclrtFree(devPtr);
        aclDestroyTensorDesc(d);
    }
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
