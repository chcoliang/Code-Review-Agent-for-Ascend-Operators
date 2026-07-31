# Code Review: test_softmax_lifecycle.cpp (A200)

## Bug 1: 循环内创建的 TensorDesc 和 DataBuffer 未释放（资源泄漏）

- **位置**: 第 10-11 行（循环体内，第 8-13 行循环）
- **类型**: 资源泄漏 / 内存泄漏
- **严重程度**: 高
- **描述**: 在 100 次循环中，每次迭代调用 `aclCreateTensorDesc` 和 `aclCreateDataBuffer` 分配资源，但循环体内和循环后均无对应的 `aclDestroyTensorDesc(desc)` 和 `aclDestroyDataBuffer(buf)` 释放调用。这导致 100 个 TensorDesc 对象和 100 个 DataBuffer 对象泄漏。在长时间运行或大规模循环场景下会耗尽 host 内存。
- **触发条件**: 每次循环执行时必然触发泄漏。循环次数越多，泄漏越严重。
- **测试方案**: 使用 Valgrind 或 AddressSanitizer 运行程序，检测 "definitely lost" 内存块数量。预期检测到 200 个未释放的分配（100 个 desc + 100 个 buf）。修复：在循环末尾添加 `aclDestroyTensorDesc(desc);` 和 `aclDestroyDataBuffer(buf);`。

## Bug 2: aclCreateDataBuffer 传入 nullptr 作为设备内存地址

- **位置**: 第 11 行
- **类型**: 逻辑错误 / 无效参数
- **严重程度**: 中
- **描述**: `aclCreateDataBuffer(nullptr, 4 * 1024 * 2)` 创建了一个数据缓冲区描述符，但设备内存指针为 nullptr。若该 buffer 被传入算子执行接口（如 `aclopExecuteV2`），将导致算子访问空指针，引发 device 端异常或 host 端崩溃。即使作为生命周期测试，也应使用有效指针或明确标注为非执行测试。
- **触发条件**: 将此 DataBuffer 用于实际算子执行时触发。
- **测试方案**: 将该 buffer 传入 softmax 算子执行，验证是否返回错误码或产生崩溃。修复：先调用 `aclrtMalloc` 分配 device 内存，再传入有效指针。

## Bug 3: 缺少 aclrtCreateStream / aclrtDestroyStream 的生命周期管理

- **位置**: 第 4-17 行（整个 main 函数）
- **类型**: 测试不完整
- **严重程度**: 低
- **描述**: 作为生命周期测试，代码仅测试了 TensorDesc 和 DataBuffer 的创建（且未释放），但未包含 stream 创建/销毁、算子执行、以及 device 内存分配/释放的完整生命周期验证。测试覆盖不充分。
- **触发条件**: N/A（测试质量问题）。
- **测试方案**: 补充完整的生命周期测试：创建 -> 使用 -> 销毁，涵盖所有 ACL 资源类型。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 10-11 | 资源泄漏 | 高 | TensorDesc/DataBuffer 循环创建未释放 |
| 2 | 11 | 无效参数 | 中 | DataBuffer 使用 nullptr 作为设备地址 |
| 3 | 4-17 | 测试不完整 | 低 | 生命周期测试缺少释放和完整覆盖 |
