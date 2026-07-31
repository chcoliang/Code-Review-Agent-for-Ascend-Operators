# A235 - test_scaled_softmax_lifecycle.cpp 代码审查报告

## Bug 1: 循环内 TensorDesc 资源泄漏

- **位置**: 第 7 行（循环体内 `aclCreateTensorDesc`）
- **类型**: 资源泄漏
- **严重程度**: 高
- **描述**: 在 `for` 循环（100次迭代）中，每次调用 `aclCreateTensorDesc` 创建张量描述符，但循环体内从未调用 `aclDestroyTensorDesc(d)` 释放。每次迭代后指针 `d` 被覆盖，前一次创建的描述符句柄丢失，累计泄漏 100 个 TensorDesc 对象。
- **触发条件**: 程序执行即触发。长期运行或大循环次数时主机内存持续增长。
- **测试方案**:
  1. 在循环前后监控进程内存（RSS），验证修复前内存持续增长、修复后保持稳定。
  2. 使用 valgrind/ASan 检测内存泄漏。

## Bug 2: 循环内 DataBuffer 资源泄漏

- **位置**: 第 8 行（循环体内 `aclCreateDataBuffer`）
- **类型**: 资源泄漏
- **严重程度**: 高
- **描述**: 每次循环调用 `aclCreateDataBuffer` 创建数据缓冲区描述符，但从未调用 `aclDestroyDataBuffer(b)` 释放。累计泄漏 100 个 DataBuffer 对象。
- **触发条件**: 程序执行即触发。
- **测试方案**:
  1. 使用 valgrind 检测泄漏，期望报告 100 个未释放的 DataBuffer。
  2. 循环次数增大到 10000 后观察内存占用变化。

## Bug 3: aclCreateDataBuffer 使用 nullptr 作为设备地址

- **位置**: 第 8 行
- **类型**: 逻辑缺陷
- **严重程度**: 中
- **描述**: `aclCreateDataBuffer(nullptr, 4*8*512*512*2)` 传入空指针作为数据地址，声明了 16MB 大小但无实际设备内存后端。若后续使用此 DataBuffer 执行算子，将导致空指针访问或设备异常。作为生命周期测试，应验证完整的 malloc->create->destroy->free 流程。
- **触发条件**: 将此 DataBuffer 传递给算子执行时触发设备错误。
- **测试方案**: 用此 DataBuffer 下发算子，验证是否报错；修改为先 aclrtMalloc 再传入有效地址。

## Bug 4: 缺少 API 返回值/空指针检查

- **位置**: 第 3、4、7、8 行
- **类型**: 错误处理缺失
- **严重程度**: 低
- **描述**: `aclCreateTensorDesc` 和 `aclCreateDataBuffer` 在失败时返回 nullptr，代码未做空指针检查。`aclInit`、`aclrtSetDevice` 返回值也未检查。
- **触发条件**: 系统资源不足导致创建失败时，后续使用 nullptr 导致崩溃。
- **测试方案**: 在资源受限环境下运行，验证空指针处理逻辑。

---

## 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 7 | 资源泄漏 | 高 | 循环内 aclCreateTensorDesc 未 Destroy，泄漏 100 个描述符 |
| 2 | 8 | 资源泄漏 | 高 | 循环内 aclCreateDataBuffer 未 Destroy，泄漏 100 个缓冲区 |
| 3 | 8 | 逻辑缺陷 | 中 | DataBuffer 使用 nullptr 地址，无实际设备内存支持 |
| 4 | 3,4,7,8 | 错误处理缺失 | 低 | API 返回值/空指针未检查 |
