# Ascend NPU 算子代码审查报告

**文件**: `test_swish_lifecycle.cpp`  
**审查范围**: 资源生命周期、内存泄漏

---

### Bug 1: aclTensorDesc 资源泄漏

- **位置**: 第 7 行，循环体内（第 5-10 行）
- **类型**: 资源泄漏（Resource Leak）
- **严重程度**: 严重（Critical）
- **描述**: 在 for 循环中每次迭代通过 `aclCreateTensorDesc` 创建 TensorDesc 对象，但循环体内及循环结束后均未调用 `aclDestroyTensorDesc(d)` 释放该资源。循环执行 100 次，将累积泄漏 100 个 TensorDesc 对象。
- **触发条件**: 程序正常执行即触发，每次循环迭代都会泄漏一个 TensorDesc。在长时间运行或大循环次数场景下，内存占用持续增长，最终可能导致 OOM。
- **修复方案**: 在循环体末尾添加 `aclDestroyTensorDesc(d);`
- **测试方案**: 使用 Valgrind/ASan 或 ACL 内部资源计数器，验证循环结束后 TensorDesc 分配数为 0；对比修复前后进程 RSS 内存增长情况。

---

### Bug 2: aclDataBuffer 资源泄漏

- **位置**: 第 8 行，循环体内（第 5-10 行）
- **类型**: 资源泄漏（Resource Leak）
- **严重程度**: 严重（Critical）
- **描述**: 在 for 循环中每次迭代通过 `aclCreateDataBuffer` 创建 DataBuffer 对象，但循环体内及循环结束后均未调用 `aclDestroyDataBuffer(b)` 释放该资源。循环 100 次累积泄漏 100 个 DataBuffer 句柄。
- **触发条件**: 程序正常执行即触发。DataBuffer 管理结构在 host 侧持续累积，若实际分配了 device 内存则更为严重。
- **修复方案**: 在循环体末尾添加 `aclDestroyDataBuffer(b);`
- **测试方案**: 同 Bug 1，使用内存检测工具验证 DataBuffer 资源全部释放；监控 device 内存使用量确认无残留。

---

### Bug 3: 缺少 aclrtStream 及实际 Device 内存分配（潜在逻辑缺陷）

- **位置**: 第 8 行
- **类型**: 逻辑缺陷 / 空指针使用风险
- **严重程度**: 中等（Medium）
- **描述**: `aclCreateDataBuffer(nullptr, 16*4096*2)` 传入 nullptr 作为数据指针，意味着 DataBuffer 并未关联有效的 device 内存。若后续将此 buffer 用于算子执行，将导致空指针访问或未定义行为。即使作为测试代码，也未体现完整的生命周期管理模式（应包含 `aclrtMalloc` / `aclrtFree`）。
- **触发条件**: 当该 DataBuffer 被传入算子执行接口时触发非法内存访问。
- **修复方案**: 使用 `aclrtMalloc` 分配 device 内存，将返回指针传入 `aclCreateDataBuffer`，并在释放 DataBuffer 后调用 `aclrtFree`。
- **测试方案**: 完善测试用例使其执行完整 Swish 算子调用，验证无段错误；检查 device 内存分配/释放配对。

---

### Bug 4: 缺少 API 返回值检查

- **位置**: 第 3、4、7、8 行
- **类型**: 错误处理缺失（Missing Error Handling）
- **严重程度**: 低（Low）
- **描述**: `aclInit`、`aclrtSetDevice`、`aclCreateTensorDesc`、`aclCreateDataBuffer` 的返回值均未检查。若任一调用失败（返回 nullptr 或非零错误码），后续操作将在无效状态下继续执行，可能引发未定义行为或掩盖真实错误。
- **触发条件**: 设备不可用、参数非法、系统资源不足等异常场景。
- **修复方案**: 对每个 ACL API 调用检查返回值，失败时打印错误信息并执行清理退出。
- **测试方案**: 模拟设备不可用场景（如设置错误设备 ID），验证程序能正确报错退出而非崩溃。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 关键描述 |
|------|------|----------|----------|----------|
| 1 | 第 7 行（循环内） | 资源泄漏 | 严重 | aclTensorDesc 未释放，泄漏 100 次 |
| 2 | 第 8 行（循环内） | 资源泄漏 | 严重 | aclDataBuffer 未释放，泄漏 100 次 |
| 3 | 第 8 行 | 逻辑缺陷 | 中等 | DataBuffer 使用 nullptr，无实际 device 内存 |
| 4 | 第 3/4/7/8 行 | 错误处理缺失 | 低 | ACL API 返回值未检查 |
