# LeakyRelu DAG 代码审查报告

文件: `leaky_relu_dag.h`

---

### Bug 1: LeakyReluDag 中标量参数类型与计算类型不匹配

- **位置**: 第 26 行
  ```cpp
  using OpLeakRelu = Bind<Vec::LeakyRelu<U>, OpCopyInX, Placeholder::Var<T, 0>>;
  ```
- **类型**: 类型错误 (Type Mismatch)
- **严重程度**: 高
- **描述**: `Vec::LeakyRelu<U>` 以类型 `U` 进行计算，期望其标量参数(negative slope)也为类型 `U`。但 `Placeholder::Var<T, 0>` 提供的标量类型为 `T`（默认 `float`）。当 `U` 为 `half`/`float16` 时，向一个 float16 向量运算传入 float 标量，导致类型不匹配，可能引发编译错误或隐式截断产生精度问题。对比 `LeakyReluCastDag` 中先将数据 Cast 到 `T` 再用 `Vec::LeakyRelu<T>` 配合 `Placeholder::Var<T, 0>`，类型是一致的。
- **触发条件**: 当模板参数 `U` 实例化为 `half`（float16）等非 float 类型时触发。
- **修复建议**: 将标量类型改为与计算类型一致：
  ```cpp
  using OpLeakRelu = Bind<Vec::LeakyRelu<U>, OpCopyInX, Placeholder::Var<U, 0>>;
  ```
  或者保持 `Var<T,0>` 但将 LeakyRelu 计算类型改为 T（但这会与输入/输出类型矛盾）。
- **测试方案**: 以 `U=half, T=float` 实例化 `LeakyReluDag`，输入 negative slope = 0.01，输入包含正负混合值，验证输出是否与 PyTorch `F.leaky_relu` 对齐；检查编译期是否有类型告警。

---

### Bug 2: LeakyReluCastDag 输出 Cast 使用了错误的舍入模式

- **位置**: 第 37-38 行
  ```cpp
  constexpr static int CAST_MODE_RINT = 1;
  using OpCopyOutCast = Bind<Vec::Cast<U, T, 0>, OpLeakRelu>;
  ```
- **类型**: 精度错误 (Precision Bug)
- **严重程度**: 中
- **描述**: 开发者定义了 `CAST_MODE_RINT = 1`（四舍五入模式），表明意图使用 round-to-nearest-integer 舍入。但实际 `Vec::Cast<U, T, 0>` 的第三个模板参数硬编码为 `0`（截断模式），并未使用 `CAST_MODE_RINT`。当从高精度 `T`(float) 转回低精度 `U`(float16) 时，截断模式会造成系统性的精度偏差（总是向零方向截断），而非更合理的就近舍入。
- **触发条件**: 当 `U=half, T=float` 时，LeakyRelu 计算结果从 float cast 回 float16，所有无法精确表示为 float16 的值都会被截断而非四舍五入。
- **修复建议**: 使用已定义的常量：
  ```cpp
  using OpCopyOutCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpLeakRelu>;
  ```
- **测试方案**: 构造输入使得 LeakyRelu 输出落在两个相邻 float16 可表示值的中间（如 0.100097...），对比截断模式和舍入模式的输出差异；使用大批量随机数据统计与 PyTorch 参考实现的平均绝对误差(MAE)。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 26 行 `LeakyReluDag::OpLeakRelu` | 类型错误 | 高 | 标量 Var<T,0>(float) 与 LeakyRelu<U>(half) 类型不匹配 |
| 2 | 第 38 行 `LeakyReluCastDag::OpCopyOutCast` | 精度错误 | 中 | 定义了 CAST_MODE_RINT=1 却使用硬编码 0（截断模式），舍入行为错误 |
