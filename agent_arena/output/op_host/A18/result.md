# Ascend NPU 算子代码审查报告 - A18 (Mul Tiling Arch35)

## Bug 列表

### Bug 1: COMPLEX64 类型映射错误 - 使用了错误的 OpDag

- **位置**: `mul_tiling_arch35.cpp` 第 154-157 行
- **类型**: 类型映射错误
- **严重程度**: 严重 (Critical)
- **描述**: `DT_COMPLEX64` 类型组合映射到 `MulOp<int64_t>::OpDag`，这与 `DT_INT64` 的映射完全相同（第 134-137 行）。`DT_COMPLEX64` 由两个 float32 组成（实部+虚部），复数乘法公式为 `(a+bi)*(c+di) = (ac-bd) + (ad+bc)i`，需要专门的复数乘法 DAG（如 `MulComplex64Op`），而非简单的标量 int64 乘法。当前实现会将 64 位复数数据直接当作 int64 整数进行标量乘法，产生完全错误的计算结果。
- **触发条件**: 当输入和输出均为 `DT_COMPLEX64` 类型时触发，执行复数乘法运算会得到错误结果。
- **测试方案**: 构造两个 COMPLEX64 张量（如 `(1+2i)` 和 `(3+4i)`），验证输出是否为正确的复数乘法结果 `(-5+10i)`，而非 int64 位模式的乘积。

---

### Bug 2: PostTiling 无条件添加 DCACHE_SIZE 导致非 Double 类型内存过度分配

- **位置**: `mul_tiling_arch35.cpp` 第 210 行
- **类型**: Tiling 参数错误
- **严重程度**: 中等 (Medium)
- **描述**: `PostTiling()` 中 `SetLocalMemorySize(static_cast<uint32_t>(ubSize_ + DCACHE_SIZE))` 无条件将 `DCACHE_SIZE`（32KB）加到 UB 大小上。然而 `DCACHE_SIZE` 作为 `extraSize` 仅在 `DT_DOUBLE` 分支（第 147 行）中使用。对于非 Double 类型，这多声明了 32KB 的 local memory，虽然不会导致计算错误，但会导致资源浪费，在 UB 空间紧张的场景下可能影响并行度或导致内存分配失败。
- **触发条件**: 所有非 Double 数据类型的 Mul 算子执行时，都会多申请 32KB 的 local memory。
- **测试方案**: 对比 float32 类型 Mul 算子实际使用的 UB 大小和声明的 local memory 大小，验证是否存在 32KB 的多余分配；在 UB 资源接近满载时测试是否影响算子调度。

---

### Bug 3: PostTiling 中 int64_t 到 uint32_t 的强制类型转换可能溢出

- **位置**: `mul_tiling_arch35.cpp` 第 210 行
- **类型**: 类型转换溢出
- **严重程度**: 低 (Low)
- **描述**: `ubSize_` 类型为 `int64_t`（见头文件第 48 行），通过 `static_cast<uint32_t>(ubSize_ + DCACHE_SIZE)` 转换为 `uint32_t`。如果平台返回的 UB 大小超过 4GB（UINT32_MAX），转换会发生截断。虽然当前 Ascend NPU 的 UB 通常远小于 4GB，但缺少溢出保护是一个潜在风险点。
- **触发条件**: 当平台报告的 UB 大小加上 DCACHE_SIZE 超过 `UINT32_MAX`（约 4GB）时触发截断。当前硬件平台下不太可能触发。
- **测试方案**: 通过 mock 平台信息设置一个极大的 ubSize 值（超过 UINT32_MAX），验证 SetLocalMemorySize 是否收到正确值或是否有错误处理。

---

## 汇总表

| 编号 | 文件 | 行号 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|----------|
| 1 | mul_tiling_arch35.cpp | 154-157 | 类型映射错误 | 严重 | COMPLEX64 错误映射为 MulOp<int64_t>，应使用复数乘法专用 DAG |
| 2 | mul_tiling_arch35.cpp | 210 | Tiling 参数错误 | 中等 | PostTiling 无条件添加 DCACHE_SIZE，非 Double 类型多分配 32KB |
| 3 | mul_tiling_arch35.cpp | 210 | 类型转换溢出 | 低 | int64_t 强制转 uint32_t 缺少溢出保护 |
