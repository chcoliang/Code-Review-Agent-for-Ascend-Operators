# Mul Tiling Arch35 代码审查报告

## Bug 列表

### Bug 1: Complex64 数据类型使用了错误的算子模板

- **位置**: 第 154-156 行
- **类型**: 逻辑错误 / 算法错误
- **严重程度**: 严重 (Critical)
- **描述**: `DT_COMPLEX64` 类型组合映射到了 `MulOp<int64_t>::OpDag`，这是一个普通的元素逐一乘法模板。然而 Complex64 由两个 float32 组成（实部+虚部），复数乘法需要遵循 `(a+bi)*(c+di) = (ac-bd)+(ad+bc)i` 的运算规则。直接将 complex64 的底层 8 字节当作 int64_t 做整数乘法，计算结果完全错误。参照第 149-152 行 `DT_COMPLEX32` 使用了专用的 `MulComplex32Op` 模板，`DT_COMPLEX64` 应当使用类似的 `MulComplex64Op` 专用复数乘法模板。
- **触发条件**: 当两个输入和输出均为 `DT_COMPLEX64` 类型时触发，计算结果将完全错误。
- **修复建议**: 将 `MulOp<int64_t>::OpDag` 替换为对应的 Complex64 专用复数乘法模板（如 `MulComplex64Op<int64_t, ...>::OpDag`）。
- **测试方案**: 构造两个 complex64 张量，如 `(1+2i)*(3+4i)`，验证结果是否为 `(-5+10i)`。使用当前代码会得到错误结果。

### Bug 2: PostTiling 无条件减去 DCACHE_SIZE 导致非 double 类型浪费 UB 空间

- **位置**: 第 209-211 行
- **类型**: 资源管理缺陷
- **严重程度**: 中等 (Medium)
- **描述**: `PostTiling()` 中通过 `SetLocalMemorySize(ubSize_ - DCACHE_SIZE)` 无条件地从可用 UB 空间中减去 32KB（DCache 大小）。然而在 `DoTiling` 的 dtype dispatch 中，只有 `DT_DOUBLE` 分支传入了 `extraSize = DCACHE_SIZE`（第 147 行），其他所有数据类型的 `extraSize` 均为默认值 0。这意味着对于非 double 类型，PostTiling 报告给框架的 UB 可用空间比实际可用空间少了 32KB，导致 tiling 策略不能充分利用 UB，可能降低性能。若 DCache 确实对所有 arch35 场景都需要预留，则 double 分支的 `extraSize = DCACHE_SIZE` 会造成重复扣减（总共 64KB），对 double 场景可用 UB 过少。
- **触发条件**: 所有非 double 数据类型的 Mul 算子执行时，UB 利用率降低；或 double 类型时 UB 被重复扣减 64KB。
- **测试方案**: 对比 float32 等类型的 tiling 结果中实际使用的 UB 大小与硬件实际可用 UB 大小，检查是否存在 32KB 的差异；对 double 类型验证 tiling 是否因 UB 不足而选择了次优分块方案。

### Bug 3: PostTiling 中 int64_t 到 uint32_t 的截断转换存在潜在溢出风险

- **位置**: 第 210 行
- **类型**: 类型安全 / 数值溢出
- **严重程度**: 低 (Low)
- **描述**: `ubSize_` 为 `int64_t` 类型，`ubSize_ - DCACHE_SIZE` 的结果通过 `static_cast<uint32_t>()` 强制转换。若由于某种异常 `ubSize_` 的值小于 `DCACHE_SIZE`（32KB），则差值为负数，转换为 `uint32_t` 后会回绕为一个极大的正整数，导致设置的 LocalMemorySize 异常。此外若平台返回的 UB 大于 4GB（理论上不会但缺乏防护），uint32_t 也会截断。
- **触发条件**: `GetPlatformInfo()` 返回异常小的 UB 大小（如 compileInfo 中 ubSize 配置错误，或 platformInfo 返回的 UB 大小经 `/2` 后小于 32KB）。
- **修复建议**: 在 `static_cast` 前添加断言或条件检查，确保 `ubSize_ > DCACHE_SIZE`。
- **测试方案**: 模拟 `ubSize_` 为 0 或小于 DCACHE_SIZE 的场景，观察 `SetLocalMemorySize` 是否设置了异常大的值。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 154-156 行 | 逻辑错误 | 严重 | Complex64 使用 `MulOp<int64_t>` 而非复数乘法专用模板，计算结果完全错误 |
| 2 | 第 209-211 行 | 资源管理 | 中等 | PostTiling 无条件减去 DCACHE_SIZE，非 double 类型浪费 UB，double 类型可能重复扣减 |
| 3 | 第 210 行 | 类型安全 | 低 | int64_t 差值转 uint32_t 无防护，异常输入时可能溢出回绕 |
