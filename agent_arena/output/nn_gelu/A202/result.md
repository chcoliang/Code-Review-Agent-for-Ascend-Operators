# GELU DAG 算子代码审查报告

## Bug 列表

### Bug 1: NEG_SQRT_EIGHT_OVER_PI 常量系数精度错误（0.04471 vs 0.044715）

- **位置**: 第 24 行 `const float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.04471;`
- **类型**: 计算逻辑/精度错误
- **严重程度**: 高
- **描述**: GELU 的近似公式要求系数为 `0.044715`，但代码中写为 `0.04471`，末尾缺少 `5`。正确值应为 `-1.595769121 * 0.044715`。该常量同时影响指数中的 x 项和 x^3 项的系数，导致整个 GELU 近似公式产生系统性偏差。计算得：正确值 ≈ -0.07135，实际值 ≈ -0.07134，相对误差约 0.01%，对大量数据累积后可能超出精度容忍范围。
- **触发条件**: 所有输入数据均受影响，输入绝对值越大偏差越明显（因 x^3 项放大）。
- **测试方案**: 对比标准 GELU 实现（PyTorch `torch.nn.functional.gelu`），在 x∈[-5,5] 范围内采样，检查最大绝对误差和相对误差是否超过 float16/float32 的 ULP 容差。

---

### Bug 2: 输出 Cast 使用 CAST_MODE_RINT 导致结果被四舍五入为整数

- **位置**: 第 77 行 `using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpLogResult>;`
- **类型**: 类型转换/精度错误
- **严重程度**: 致命
- **描述**: GELU 的输出是连续浮点值（范围约 [-0.17, +∞)），将 float 结果 Cast 回原始类型 U（如 half/bfloat16）时使用了 `CAST_MODE_RINT`（round to nearest integer）。这会将所有输出值四舍五入为整数，例如 GELU(0.5)≈0.346 会被舍入为 0，GELU(1.0)≈0.841 会被舍入为 1。这完全破坏了 GELU 的数学语义。应使用 `CAST_MODE_NONE` 进行普通精度转换。
- **触发条件**: 当模板参数 U != float（即 U 为 half 或 bfloat16）时必现。所有非整数输出均被错误舍入。
- **测试方案**: 以 half 类型输入 x=0.5，期望输出 ≈0.346；若输出为 0 则确认此 bug。对比 `CAST_MODE_NONE` 和 `CAST_MODE_RINT` 的输出差异。

---

### Bug 3: 循环内 Mask 未按剩余元素数更新，最后一次迭代可能越界

- **位置**: 第 50 行 `mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);`
- **类型**: 内存访问/计算逻辑错误
- **严重程度**: 中
- **描述**: Mask 在循环中始终以总元素数 `count` 计算，而非当前迭代的剩余元素数。当 `count` 不是向量长度 `vl` 的整数倍时，最后一次迭代应仅处理 `count - loopIdx * vl` 个元素。当前实现中，若 `count > vl`，`UpdateMask(count)` 很可能生成全 1 掩码，导致最后一次迭代读写超出有效数据范围的内存，可能产生垃圾数据写入或读取未初始化内存。正确写法应为：`mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(min(vl, count - loopIdx * vl));` 或在循环外计算完整掩码、循环末尾计算尾部掩码。
- **触发条件**: 当输入元素总数 `count` 不是 `VECTOR_REG_WIDTH / sizeof(T)` 的整数倍时触发。
- **测试方案**: 构造 count = vl + 1 的测试用例（如 float 时 count=65，vl=64），检查输出 buffer 第 66~128 位置是否被错误写入；检查输出第 65 个元素是否为正确的 GELU 值。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 24 行 | 精度/计算逻辑 | 高 | 常量 0.04471 应为 0.044715，导致 GELU 系统性精度偏差 |
| 2 | 第 77 行 | 类型转换 | 致命 | 输出 Cast 使用 RINT 模式将浮点结果舍入为整数，完全破坏结果 |
| 3 | 第 50 行 | 内存访问/逻辑 | 中 | Mask 未按迭代更新，尾部迭代越界读写 |
