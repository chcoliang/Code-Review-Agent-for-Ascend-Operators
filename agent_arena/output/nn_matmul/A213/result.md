# A213 - test_matmul_stress.cpp 代码审查报告

## Bug 1: 循环内分配内存未释放（严重内存泄漏）

**位置**: 第 7-11 行

**问题描述**: 在 `for` 循环中每次迭代都通过 `aclrtMalloc` 分配 32MB 设备内存，但循环体内没有调用 `aclrtFree(ws)` 释放。循环执行 500 次，累计泄漏 500 * 32MB = 16GB 设备内存，极易导致 NPU 设备 OOM。

**问题代码**:
```cpp
for (int i = 0; i < 500; i++) {
    void* ws = nullptr;
    aclrtMalloc(&ws, 32*1024*1024, ACL_MEM_MALLOC_HUGE_FIRST);
    // 缺少 aclrtFree(ws);
}
```

**修复建议**:
```cpp
for (int i = 0; i < 500; i++) {
    void* ws = nullptr;
    aclrtMalloc(&ws, 32*1024*1024, ACL_MEM_MALLOC_HUGE_FIRST);
    // ... 使用 ws 进行计算 ...
    aclrtFree(ws);
}
```

**严重程度**: 严重

---

## Bug 2: 缺少 aclrtMalloc 返回值检查

**位置**: 第 9 行

**问题描述**: 未检查 `aclrtMalloc` 的返回值。在压力测试场景下（500次连续分配），后续分配很可能因设备内存耗尽而失败，使用未分配成功的指针会导致未定义行为。

**修复建议**:
```cpp
aclError ret = aclrtMalloc(&ws, 32*1024*1024, ACL_MEM_MALLOC_HUGE_FIRST);
if (ret != ACL_SUCCESS) {
    // 错误处理
    break;
}
```

**严重程度**: 中等

---

## 汇总表

| 编号 | Bug 描述 | 位置 | 严重程度 | 类别 |
|------|----------|------|----------|------|
| 1 | 循环内 aclrtMalloc 未释放，泄漏 16GB 设备内存 | 第 7-11 行 | 严重 | 内存泄漏 |
| 2 | 未检查 aclrtMalloc 返回值 | 第 9 行 | 中等 | 错误处理 |
