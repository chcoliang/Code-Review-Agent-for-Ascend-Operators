# A214 - test_matmul_pipeline.cpp 代码审查报告

## Bug 1: 异步操作未同步即读取结果（数据竞争）

**位置**: 第 14 行

**问题描述**: MatMul 算子在 stream 上异步执行（第 11 行注释表明），但在 `aclrtMemcpy` 读取结果之前没有调用 `aclrtSynchronizeStream(stream)` 进行同步。这意味着设备端计算可能尚未完成，主机端就开始读取输出缓冲区 `c` 的数据，导致读到未计算完成的脏数据或未初始化数据。

**问题代码**:
```cpp
// launch MatMul on stream...

float h[1024];
aclrtMemcpy(h, sizeof(h), c, sizeof(h), ACL_MEMCPY_DEVICE_TO_HOST);
```

**修复建议**:
```cpp
// launch MatMul on stream...
aclrtSynchronizeStream(stream);

float h[1024];
aclrtMemcpy(h, sizeof(h), c, sizeof(h), ACL_MEMCPY_DEVICE_TO_HOST);
```

**严重程度**: 严重

---

## Bug 2: 同步 memcpy 与异步 stream 操作存在流水线阻塞问题

**位置**: 第 14 行

**问题描述**: 作为 pipeline 测试，使用同步的 `aclrtMemcpy` 而非异步的 `aclrtMemcpyAsync` 会阻塞 host 端，无法实现真正的流水线并行。但更根本的问题是即使使用同步 memcpy，也必须在其之前确保 stream 上的计算已完成。

**严重程度**: 中等

---

## Bug 3: 缺少 API 返回值检查

**位置**: 第 8-10、14 行

**问题描述**: `aclrtMalloc` 和 `aclrtMemcpy` 均未检查返回值。若内存分配失败，后续使用空指针会导致程序崩溃。

**严重程度**: 中等

---

## 汇总表

| 编号 | Bug 描述 | 位置 | 严重程度 | 类别 |
|------|----------|------|----------|------|
| 1 | 异步计算未同步即读取结果，数据竞争 | 第 14 行 | 严重 | 同步问题 |
| 2 | 同步 memcpy 破坏流水线并行性 | 第 14 行 | 中等 | 设计问题 |
| 3 | 缺少 API 返回值检查 | 第 8-10、14 行 | 中等 | 错误处理 |
