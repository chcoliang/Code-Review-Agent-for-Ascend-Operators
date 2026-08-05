# Ascend NPU 算子代码审查报告 - gelu_dag.h (A46)

## Bug 列表

### Bug 1: CopyOut 类型映射错误

- **位置**: 第78行
  ```cpp
  using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResultCast>;
  ```
- **类型**: 类型映射错误 (Type Mismatch)
- **严重程度**: 严重 (Critical)
- **描述**: `OpResultCast` 将数据从 `T`(float) 转换为 `U`(外部类型，如 half/bfloat16)，其输出类型为 `U`。但 `CopyOut<T>` 和 `Placeholder::Out0<T>` 均声明为类型 `T`(float），与 `OpResultCast` 的输出类型 `U` 不匹配。应改为 `Vec::CopyOut<U>` 和 `Placeholder::Out0<U>`。
- **触发条件**: 当模板参数 `U` 与 `T` 不同时（即 `U` 为 half 或 bfloat16 而 `T` 为 float），CopyOut 阶段类型不匹配，导致编译错误或运行时写出错误数据/内存越界。
- **修复建议**:
  ```cpp
  using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;
  ```
- **测试方案**:
  1. 使用 `U=half, T=float` 实例化 `GeluDAG`，验证编译是否通过。
  2. 对比输出 tensor 的 dtype 是否与输入一致（half in -> half out）。
  3. 使用随机输入数据，对比 CPU 参考实现结果，检验精度。

---

### Bug 2: Cast 模式使用 CAST_MODE_RINT 错误

- **位置**: 第77行
  ```cpp
  using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpLogResult>;
  ```
- **类型**: 逻辑错误 (Logic Error)
- **严重程度**: 高 (High)
- **描述**: `CAST_MODE_RINT` 表示"四舍五入到最近整数"的舍入模式，适用于浮点到整数的转换。此处是将 float 结果转回 half/bfloat16（浮点到浮点的精度降低），应使用 `CAST_MODE_NONE`（默认截断/就近舍入到偶数的 IEEE 标准行为）。使用 RINT 会导致所有 GELU 输出值被错误地舍入为整数值（如 0.841 -> 1.0），完全破坏计算精度。
- **触发条件**: 任何非整数的 GELU 输出都会被舍入为最近整数，即几乎所有输入都会产生错误结果。当 `U != T` 时（如 U=half）此 Cast 节点被激活，bug 即触发。
- **修复建议**:
  ```cpp
  using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_NONE>, OpLogResult>;
  ```
- **测试方案**:
  1. 输入 `x = 0.5`（half），期望输出约 `0.3457`，检查是否被错误舍入为 `0.0`。
  2. 输入 `x = 1.0`（half），期望输出约 `0.8413`，检查是否被错误舍入为 `1.0`。
  3. 批量随机测试，统计与 CPU 参考实现的最大绝对误差，RINT 模式下误差应远大于阈值。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第78行 `OpCopyOut` | 类型映射错误 | 严重 | CopyOut 和 Placeholder 使用了内部计算类型 `T` 而非外部类型 `U`，与 OpResultCast 输出类型不匹配 |
| 2 | 第77行 `OpResultCast` | 逻辑错误 | 高 | float->half 类型转换错误使用 CAST_MODE_RINT，导致结果被舍入为整数 |
