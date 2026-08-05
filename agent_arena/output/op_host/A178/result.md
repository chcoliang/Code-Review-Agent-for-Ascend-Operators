# Mul Tiling Arch35 代码审查报告

## Bug 列表

### Bug 1: Complex64 使用了错误的 OpDag 模板类型

- **位置**: 第 154-157 行
- **类型**: 逻辑错误 / 语义错误
- **严重程度**: 严重 (Critical)
- **描述**: `DT_COMPLEX64` 类型（即 `std::complex<float>`，8字节）的分支使用了 `MulOp<int64_t>::OpDag` 进行计算。`int64_t` 虽然与 `complex64` 大小相同（8字节），但复数乘法 `(a+bi)*(c+di) = (ac-bd)+(ad+bc)i` 的语义完全不同于整数乘法。这会导致 Complex64 乘法结果完全错误。对比第 149-153 行的 `DT_COMPLEX32` 使用了专用的 `MulComplex32Op`，此处应使用类似的 `MulComplex64Op` 专用模板。
- **触发条件**: 当输入输出均为 `DT_COMPLEX64` 类型时，所有 Mul 计算结果均错误。
- **测试方案**: 构造两个 Complex64 张量，如 `(1+2i)*(3+4i)`，预期结果为 `(-5+10i)`，实际会得到错误的整数乘法结果。

### Bug 2: PostTiling 中 DCACHE_SIZE 对所有数据类型无条件扣除，与 double 分支 extraSize 存在双重扣除

- **位置**: 第 210 行 与 第 147 行
- **类型**: 逻辑错误 / 资源计算错误
- **严重程度**: 中等 (Medium)
- **描述**: `PostTiling()` 中对所有数据类型无条件执行 `SetLocalMemorySize(ubSize_ - DCACHE_SIZE)`，即始终预留 128KB 的 DCache 空间。而 double 分支（第147行）在调用 `DoTiling` 时额外传入了 `extraSize = DCACHE_SIZE`。这意味着 double 分支被双重扣除了 DCACHE_SIZE 空间：一次在 DoTiling 内部通过 extraSize 预留，一次在 PostTiling 中通过减法预留。对于非 double 类型，即使不需要 DCache 预留，也被无条件扣除了 128KB，造成 UB 空间浪费。
- **触发条件**: (1) double 类型运算时，实际可用 UB 比预期少 128KB，可能导致 tiling 效率下降或大 shape 场景失败；(2) 非 double 类型浪费 128KB UB 空间。
- **测试方案**: 使用 double 类型大 shape 输入，验证 tiling 分块是否正确利用了所有可用 UB 空间；对比 float 类型相同 shape 的性能，检查是否有不必要的性能损失。

### Bug 3: PostTiling 中整数下溢和 uint32_t 截断风险

- **位置**: 第 210 行
- **类型**: 数值溢出 / 类型转换错误
- **严重程度**: 中等 (Medium)
- **描述**: `static_cast<uint32_t>(ubSize_ - DCACHE_SIZE)` 存在两个风险：(1) 若 `ubSize_` 小于 `DCACHE_SIZE`（131072字节），减法结果为负数，转为 `uint32_t` 后变成极大值，导致内存越界；(2) 若 `ubSize_ - DCACHE_SIZE` 超过 `UINT32_MAX`（约4GB），发生截断。虽然当前硬件 UB 通常大于 128KB 且不超过 4GB，但缺乏防御性检查。
- **触发条件**: (1) 平台信息获取异常导致 `ubSize_` 为异常小值；(2) `GetPlatformInfo()` 未被调用时 `ubSize_` 未初始化（取决于类成员默认值）。
- **测试方案**: Mock `GetPlatformInfo()` 返回极小的 ubSize（如 0 或 64KB），检查 `PostTiling` 是否产生异常行为或崩溃。

### Bug 4: ubSize_ 成员变量可能未初始化即被使用

- **位置**: 第 210 行（使用处）、第 214-229 行（初始化处）
- **类型**: 未初始化变量
- **严重程度**: 中等 (Medium)
- **描述**: `PostTiling()` 依赖 `ubSize_` 的值，但 `ubSize_` 的初始化在 `GetPlatformInfo()` 中完成。如果调用顺序异常（`PostTiling` 先于 `GetPlatformInfo` 执行），或 `GetPlatformInfo` 中两个分支都未成功赋值（如 platformInfo 为 null 且 compileInfoPtr 也为 null 时提前 return），`ubSize_` 可能处于未初始化状态。虽然头文件中可能有默认值，但在当前文件中无法确认。
- **触发条件**: `GetPlatformInfo()` 中 platformInfo 为 null 且 compileInfoPtr 也为 null，函数提前返回 `GRAPH_FAILED` 后，若调用流程仍继续到 `PostTiling()`。
- **测试方案**: 构造 compileInfo 为 null 的场景，验证整体流程是否正确中止，或是否会带着未初始化值进入 PostTiling。

### Bug 5: OP_LOGD 格式说明符与 int64_t 类型不匹配

- **位置**: 第 220 行、第 226 行
- **类型**: 格式字符串错误
- **严重程度**: 低 (Low)
- **描述**: `ubSize_` 为 `int64_t` 类型，日志中使用 `%ld` 格式说明符。在 LP64 模型（Linux 64位）下 `%ld` 可匹配 `int64_t`，但在 LLP64 模型（Windows 64位）下 `long` 为 32 位，应使用 `PRId64` 宏或 `%lld` 以确保跨平台兼容性。
- **触发条件**: 在 Windows 或其他 LLP64 平台上编译运行时，日志输出值可能截断或乱码。
- **测试方案**: 在 LLP64 平台（如 Windows）上编译并检查日志输出是否正确显示 ubSize 值。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | L154-157 | 逻辑错误 | 严重 | Complex64 错误使用 MulOp<int64_t> 代替专用复数乘法模板 |
| 2 | L210, L147 | 逻辑错误 | 中等 | DCACHE_SIZE 对 double 双重扣除，对非 double 类型不必要扣除 |
| 3 | L210 | 数值溢出 | 中等 | ubSize_ - DCACHE_SIZE 下溢后 cast 为 uint32_t 产生极大值 |
| 4 | L210, L214-229 | 未初始化变量 | 中等 | ubSize_ 可能在 GetPlatformInfo 失败后仍被 PostTiling 使用 |
| 5 | L220, L226 | 格式字符串 | 低 | %ld 与 int64_t 在 LLP64 平台上不兼容 |
