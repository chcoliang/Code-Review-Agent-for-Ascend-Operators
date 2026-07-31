# A215 - test_matmul_lifecycle.cpp 代码审查报告

## Bug 1: aclTensorDesc 未销毁（资源泄漏）

**位置**: 第 7-10 行

**问题描述**: 在循环中每次迭代通过 `aclCreateTensorDesc` 创建 TensorDesc 对象，但循环体内没有调用 `aclDestroyTensorDesc(d)` 销毁。循环 100 次累计泄漏 100 个 TensorDesc 对象，造成主机端内存泄漏。

**问题代码**:
```cpp
for (int i = 0; i < 100; i++) {
    int64_t s[] = {2048, 2048};
    aclTensorDesc* d = aclCreateTensorDesc(ACL_FLOAT16, 2, s, ACL_FORMAT_ND);
    aclDataBuffer* b = aclCreateDataBuffer(nullptr, 2048*2048*2);
    // 缺少 aclDestroyTensorDesc(d);
    // 缺少 aclDestroyDataBuffer(b);
}
```

**修复建议**:
```cpp
for (int i = 0; i < 100; i++) {
    int64_t s[] = {2048, 2048};
    aclTensorDesc* d = aclCreateTensorDesc(ACL_FLOAT16, 2, s, ACL_FORMAT_ND);
    aclDataBuffer* b = aclCreateDataBuffer(nullptr, 2048*2048*2);
    // ... 使用 d 和 b ...
    aclDestroyTensorDesc(d);
    aclDestroyDataBuffer(b);
}
```

**严重程度**: 严重

---

## Bug 2: aclDataBuffer 未销毁（资源泄漏）

**位置**: 第 8-10 行

**问题描述**: 在循环中每次迭代通过 `aclCreateDataBuffer` 创建 DataBuffer 对象，但没有调用 `aclDestroyDataBuffer(b)` 销毁。100 次循环累计泄漏 100 个 DataBuffer 对象。

**严重程度**: 严重

---

## Bug 3: aclCreateDataBuffer 传入 nullptr 指针

**位置**: 第 8 行

**问题描述**: `aclCreateDataBuffer(nullptr, 2048*2048*2)` 使用 nullptr 作为数据指针创建了一个无效的 DataBuffer。正常使用时应先通过 `aclrtMalloc` 分配设备内存，再将得到的指针传入。使用 nullptr 创建的 DataBuffer 无法用于实际算子计算。

**修复建议**:
```cpp
void* devPtr = nullptr;
aclrtMalloc(&devPtr, 2048*2048*2, ACL_MEM_MALLOC_HUGE_FIRST);
aclDataBuffer* b = aclCreateDataBuffer(devPtr, 2048*2048*2);
```

**严重程度**: 中等

---

## Bug 4: 生命周期管理缺陷 - 缺少 stream 创建与销毁

**位置**: 全局

**问题描述**: 作为生命周期测试，代码没有创建 stream 来执行算子，也没有完整的算子执行流程，TensorDesc 和 DataBuffer 在创建后没有被使用就应被销毁，说明生命周期管理测试不完整。

**严重程度**: 低

---

## 汇总表

| 编号 | Bug 描述 | 位置 | 严重程度 | 类别 |
|------|----------|------|----------|------|
| 1 | aclTensorDesc 循环内创建未销毁 | 第 7-10 行 | 严重 | 生命周期管理/资源泄漏 |
| 2 | aclDataBuffer 循环内创建未销毁 | 第 8-10 行 | 严重 | 生命周期管理/资源泄漏 |
| 3 | aclCreateDataBuffer 传入 nullptr 无效指针 | 第 8 行 | 中等 | 资源管理 |
| 4 | 生命周期测试缺少 stream 及完整执行流程 | 全局 | 低 | 设计问题 |
