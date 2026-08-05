# Mul Tiling Arch35 代码审查报告

## Bug 列表

### Bug 1: Complex64 使用错误的 Tiling DAG 模板

- **位置**: 第 154-157 行
- **类型**: 逻辑错误 / Tiling 参数错误
- **严重程度**: 高
- **描述**: `DT_COMPLEX64` 类型映射到 `MulOp<int64_t>::OpDag`，这是一个普通整数乘法的 DAG。Complex64 表示每个元素由两个 float32 组成(实部+虚部)，需要使用专用的复数乘法 DAG（如 `MulComplex64Op`）来正确实现 `(a+bi)*(c+di)` 运算。当前实现会将 8 字节的复数数据当作单个 int64_t 进行整数乘法，产生完全错误的计算结果。对比第 149-153 行的 `DT_COMPLEX32` 使用了专门的 `MulComplex32Op<int32_t, int64_t>::OpDag`，可确认 Complex64 此处存在遗漏。
- **触发条件**: 当输入和输出均为 `DT_COMPLEX64` 类型时触发
- **测试方案**: 构造两个 Complex64 张量（如 `(1+2i)*(3+4i)`），验证输出是否为正确的复数乘法结果 `(-5+10i)`，而非错误的整数乘积

---

### Bug 2: PostTiling 中 LocalMemorySize 计算存在 Off-by-One 错误

- **位置**: 第 210 行
- **类型**: 计算错误 / Tiling 参数错误
- **严重程度**: 中
- **描述**: `context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE - 1))` 中存在多余的 `-1`。标准做法是将可用 UB 大小设置为 `ubSize_ - DCACHE_SIZE`（减去 D-Cache 保留区域）。多余的 `-1` 使得可用内存少了 1 字节，在对齐计算时可能导致 tiling 分块策略产生偏差，降低计算效率。正确写法应为 `ubSize_ - DCACHE_SIZE`。
- **触发条件**: 所有数据类型的 Mul 算子均会触发
- **测试方案**: 对比修复前后 tiling 分块结果，在 UB 大小恰好为对齐边界值时验证是否产生不同的分块策略

---

### Bug 3: PostTiling 中缺少 ubSize 下溢保护导致潜在的 uint32_t 溢出

- **位置**: 第 210 行
- **类型**: 数值溢出 / 鲁棒性缺陷
- **严重程度**: 高
- **描述**: 当 `ubSize_` <= `DCACHE_SIZE`（32KB）时，`ubSize_ - DCACHE_SIZE - 1` 产生负值，经 `static_cast<uint32_t>` 转换后变为一个极大的无符号整数（接近 4GB），作为 LocalMemorySize 传入会导致后续 tiling 计算使用错误的超大内存值，可能引发越界访问或内核崩溃。应在计算前添加 `ubSize_ > DCACHE_SIZE` 的检查。
- **触发条件**: 当平台返回的 UB 大小小于或等于 32KB 时（异常配置或 compileInfo 错误赋值场景）
- **测试方案**: Mock 一个 `ubSize_` 为 32KB 或更小的平台信息，验证是否产生异常的 LocalMemorySize 或程序崩溃

---

### Bug 4: Double 类型 DCACHE_SIZE 被重复扣减

- **位置**: 第 144-148 行 与 第 210 行
- **类型**: 逻辑错误 / Tiling 参数错误
- **严重程度**: 中
- **描述**: Double 分支在 `DoTiling` 调用时传入 `extraSize = DCACHE_SIZE`（第 147 行），表示需要额外预留 32KB 的 D-Cache 空间。然而 `PostTiling()`（第 210 行）对所有类型统一执行了 `ubSize_ - DCACHE_SIZE - 1`，即已经全局扣减了 DCACHE_SIZE。这导致 Double 类型实际可用 UB 被减少了两倍 DCACHE_SIZE（64KB），tiling 分块偏小，降低了计算效率。非 Double 类型则不需要 D-Cache 预留却被强制扣减，同样损失了 32KB 可用空间。
- **触发条件**: 使用 `DT_DOUBLE` 类型数据时影响最大；其他类型也有不必要的内存缩减
- **测试方案**: 对 Double 类型数据，打印实际 tiling 使用的 UB 大小，验证是否比预期少了 32KB；对比 Float 类型验证非 Double 分支是否不应扣减 DCACHE_SIZE

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 154-157 行 | 逻辑错误 | 高 | Complex64 使用 `MulOp<int64_t>` 而非专用复数乘法 DAG，计算结果完全错误 |
| 2 | 第 210 行 | 计算错误 | 中 | `SetLocalMemorySize` 多减了 1 字节（off-by-one） |
| 3 | 第 210 行 | 数值溢出 | 高 | 无 ubSize 下溢保护，可能产生 uint32_t 溢出为极大值 |
| 4 | 第 147/210 行 | 逻辑错误 | 中 | Double 类型 DCACHE_SIZE 被 extraSize 和 PostTiling 重复扣减 |
