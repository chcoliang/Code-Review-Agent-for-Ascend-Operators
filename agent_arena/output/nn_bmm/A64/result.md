# Ascend NPU Batch MatMul V3 代码审查报告

## Bug 列表

### Bug 1: cGlobal_ 类型声明错误 (GlobalTensor类型与实际数据类型不匹配)

- **位置**: 第 57 行, `BatchMatMulMultiBatchKernel` 类成员声明
- **类型**: 类型错误 (Type Mismatch)
- **严重程度**: 严重 (Critical)
- **描述**: `cGlobal_` 被声明为 `GlobalTensor<A_T>`，但它存储的是输出矩阵 C 的数据，正确类型应为 `GlobalTensor<C_T>`。对比同文件中 `BatchMatMulCommonKernel`（第 299 行）和 `BatchMatMulMultiBatchFullLoadKernel`（第 867 行）均正确使用了 `GlobalTensor<C_T>`。当 A_T 与 C_T 类型不同时（如输入 half/int8，输出 float），会导致：1) 编译错误（SetGlobalBuffer 传入 `__gm__ C_T*` 与模板参数 A_T 不符）；2) 若通过隐式转换编译通过，将以错误的元素大小访问全局内存，造成数据损坏或越界。
- **触发条件**: 当 A_TYPE::T 与 C_TYPE::T 不同时触发，例如 A 为 half/int8_t 而 C 为 float 的标准 BMM 场景。
- **修复方案**: 将第 57 行 `GlobalTensor<A_T> cGlobal_;` 改为 `GlobalTensor<C_T> cGlobal_;`
- **测试方案**: 使用 A=half, B=half, C=float 的模板参数实例化 `BatchMatMulMultiBatchKernel`，验证编译通过且输出结果正确。

---

### Bug 2: UpdateGlobalTensor 调用 CalculateabGM 缺少参数

- **位置**: 第 659 行, `BatchMatMulUnalignedMultiBatchKernel::UpdateGlobalTensor` 函数
- **类型**: 接口调用错误 (Missing Argument)
- **严重程度**: 严重 (Critical)
- **描述**: `CalculateabGM` 函数签名（第 492-493 行声明，第 578 行定义）需要 7 个参数，最后一个为 `uint64_t c0Size`。但在 `UpdateGlobalTensor`（第 659 行）中调用时只传了 6 个参数，缺少 `c0Size`。这会导致编译失败。
- **触发条件**: 任何调用 `UpdateGlobalTensor` 的路径。
- **修复方案**: 在调用处补充 c0Size 参数。由于 Init 中已经通过 `GetSizeC0<A_T>(c0Size)` 获取了该值，建议将 c0Size 保存为类成员变量（如在 `UnAlignedKernelParams` 中增加 `uint64_t c0Size` 字段），然后在第 659 行传入：`CalculateabGM(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM, innerParams_.c0Size);`
- **测试方案**: 编译包含 UpdateGlobalTensor 调用路径的算子，确认编译通过；运行多 batch 分块场景验证 workspace 地址计算正确。

---

### Bug 3: BatchMatMulUnalignedKernel 中 mm_ 成员忽略模板配置参数

- **位置**: 第 111 行, `BatchMatMulUnalignedKernel` 类成员声明
- **类型**: 逻辑错误 (Template Parameter Ignored)
- **严重程度**: 中等 (Medium)
- **描述**: 类模板接受 `MM_CFG` 配置参数，但内部 `mm_` 成员硬编码使用 `MM_CFG_NO_PRELOAD`：
  ```cpp
  MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MatmulBaseBlock, MM_CFG_NO_PRELOAD> mm_;
  ```
  同时 `BLOCK_TYPE` 模板参数也被忽略，硬编码为 `MatmulBaseBlock`。这使得用户传入的 MM_CFG 和 BLOCK_TYPE 参数完全无效，无法定制 matmul 行为。
- **触发条件**: 当用户以非 `MM_CFG_NO_PRELOAD` 的配置实例化 `BatchMatMulUnalignedKernel` 时，预期行为（如 preload 使能）不会生效。
- **修复方案**: 将第 111 行改为 `MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mm_;`
- **测试方案**: 以不同 MM_CFG（如启用 preload）实例化该 kernel，验证配置确实生效。

---

### Bug 4: Process 中 iC 计算缺少 iC4 变量（与其他维度计算模式不一致）

- **位置**: 第 170 行, `BatchMatMulUnalignedKernel::Process` 函数
- **类型**: 逻辑错误 (Inconsistent Index Calculation)
- **严重程度**: 低 (Low)
- **描述**: 在 4 层循环中，iA 和 iB 的计算采用 `iX1 + iX2 + iX3 + iX4` 模式，其中 iX4 是条件赋值的变量。但 iC 的计算为 `iC1 + iC2 + iC3 + i4`，直接使用循环变量 `i4` 而非定义一个 `iC4` 变量。由于 C 矩阵不做 broadcast（batchC4 总等于 max(batchA4, batchB4)），`iC4 = i4` 在逻辑上是正确的。但代码风格不一致，且若未来 C 维度也需要广播支持则会出错。
- **触发条件**: 当前逻辑正确，不会触发计算错误。但代码可维护性差。
- **修复方案**: 建议添加 `uint64_t iC4 = i4;` 并使用 `iC = iC1 + iC2 + iC3 + iC4;` 保持一致性。
- **测试方案**: 代码审查确认风格一致性；功能不受影响无需额外测试。

---

### Bug 5: Init 函数中计算的局部变量未使用 (Dead Code)

- **位置**: 第 900-901 行, `BatchMatMulMultiBatchFullLoadKernel::Init` 函数
- **类型**: 冗余代码 (Dead Code)
- **严重程度**: 低 (Low)
- **描述**: 变量 `nAligned` 和 `dAligned` 被计算但从未使用：
  ```cpp
  uint64_t nAligned = MMV3CeilAlign(...M, ALIGNED_H);
  uint64_t dAligned = MMV3CeilAlign(...Ka, c0Size);
  ```
  这些变量可能是早期版本用于 buffer 大小计算的残留代码，当前由 `block_.params_.singleASizeL1` 替代（第 902 行）。
- **触发条件**: 不影响运行，但增加代码理解难度，编译器可能产生 unused variable 警告。
- **修复方案**: 删除第 900-901 行。
- **测试方案**: 编译验证无警告，功能回归测试通过。

---

## 汇总表

| # | 位置 | 类型 | 严重程度 | 简述 |
|---|------|------|----------|------|
| 1 | 第 57 行 | 类型错误 | 严重 | `cGlobal_` 声明为 `GlobalTensor<A_T>` 应为 `GlobalTensor<C_T>` |
| 2 | 第 659 行 | 接口调用错误 | 严重 | `CalculateabGM` 调用缺少 `c0Size` 参数 |
| 3 | 第 111 行 | 逻辑错误 | 中等 | `mm_` 硬编码 `MM_CFG_NO_PRELOAD`，忽略模板参数 `MM_CFG` |
| 4 | 第 170 行 | 逻辑错误 | 低 | iC 计算风格与 iA/iB 不一致（功能正确但可维护性差） |
| 5 | 第 900-901 行 | 冗余代码 | 低 | `nAligned`/`dAligned` 计算后未使用 |
