# LeakyRelu DAG 代码审查报告

**文件**: `leaky_relu_dag.h`  
**平台**: Ascend 910B  

---

## Bug 1: LeakyReluDag 中负斜率标量类型与计算类型不匹配

- **位置**: 第 26 行
- **类型**: 类型不匹配 / 精度问题
- **严重程度**: 高
- **描述**: 在 `LeakyReluDag` 中，`Vec::LeakyRelu<U>` 以类型 `U` 进行计算，但负斜率标量参数使用 `Placeholder::Var<T, 0>`，其类型为 `T`（默认 `float`）。当 `U` 不等于 `T` 时（例如 `U=half, T=float`），标量类型与算子计算类型不一致，可能导致编译错误或隐式截断造成精度损失。LeakyRelu 算子要求标量参数类型与运算 tensor 类型一致。
- **触发条件**: 当实例化 `LeakyReluDag<half>` 或 `LeakyReluDag<half, float>` 时，即 `U != T` 的情况。
- **修复建议**: 将第 26 行的 `Placeholder::Var<T, 0>` 改为 `Placeholder::Var<U, 0>`，使标量类型与 LeakyRelu 的计算类型一致。
- **测试方案**: 使用 `half` 类型实例化 `LeakyReluDag<half>`，设置 negativeSlope 为 0.01，输入包含负值的 tensor，对比 CPU float 参考实现的结果，检查精度偏差是否超出允许范围。

---

## Bug 2: CAST_MODE_RINT 常量值错误（应为舍入模式但实际为截断模式）

- **位置**: 第 37-38 行
- **类型**: 精度问题
- **严重程度**: 高
- **描述**: `constexpr static int CAST_MODE_RINT = 0;` 将舍入模式命名为 `RINT`（Round to nearest Integer），但赋值为 `0`。在 Ascend C 的 Cast API 中，mode=0 表示截断（向零取整），mode=1 表示四舍五入到最近偶数（rint）。这导致从浮点类型 `T` 转换回整数类型 `U`（如 int8/int32）时使用截断而非期望的四舍五入，产生精度偏差。
- **触发条件**: 当 `LeakyReluCastDag` 用于整数类型（如 `U=int8_t, T=float`）时，输出 Cast 回整数会截断而非舍入，例如 0.7 会变成 0 而非 1。
- **修复建议**: 将第 37 行改为 `constexpr static int CAST_MODE_RINT = 1;`，使用正确的 rint 舍入模式。
- **测试方案**: 实例化 `LeakyReluCastDag<int8_t, float>`，输入值使 LeakyRelu 输出为非整数（如输入 -3，negativeSlope=0.1，期望输出 -0.3 → 舍入到 0），验证输出是否为四舍五入结果而非截断结果。

---

## Bug 3: LeakyReluDag 缺少对不支持 LeakyRelu 类型的编译期保护

- **位置**: 第 23-30 行
- **类型**: DAG 逻辑 / 鲁棒性
- **严重程度**: 低
- **描述**: `LeakyReluDag` 直接对类型 `U` 调用 `Vec::LeakyRelu<U>`，但 Ascend 910B 上 LeakyRelu 仅支持 `half` 和 `float` 类型。若用户误用整数类型（如 `int32_t`）实例化此模板而非使用 `LeakyReluCastDag`，将产生难以理解的编译错误，缺少 `static_assert` 进行友好的编译期类型检查。
- **触发条件**: 使用非 float/half 类型直接实例化 `LeakyReluDag`，如 `LeakyReluDag<int8_t>`。
- **修复建议**: 在 `LeakyReluDag` 中添加 `static_assert` 限制 `U` 只能为 `half` 或 `float`。
- **测试方案**: 尝试用 `int8_t`、`int32_t` 等类型实例化 `LeakyReluDag`，验证是否给出清晰的编译错误信息。

---

## 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 26 | 类型不匹配 | 高 | LeakyReluDag 中标量参数类型 T 与计算类型 U 不一致 |
| 2 | 37-38 | 精度问题 | 高 | CAST_MODE_RINT 值为 0（截断），应为 1（四舍五入） |
| 3 | 23-30 | DAG 逻辑/鲁棒性 | 低 | 缺少对 U 类型的 static_assert 编译期保护 |
