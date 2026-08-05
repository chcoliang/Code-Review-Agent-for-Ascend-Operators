# LeakyReLU DAG 代码审查报告

## 审查文件
`leaky_relu_dag.h`

---

### Bug 1: CAST_MODE_RINT 值错误

- **位置**: 第 37 行 `constexpr static int CAST_MODE_RINT = 0;`
- **类型**: 逻辑错误 / 常量值错误
- **严重程度**: 高
- **描述**: 变量命名为 `CAST_MODE_RINT`（表示四舍五入到最近整数），但赋值为 `0`。在 AscendC 中，Cast 操作的舍入模式定义为：`0` = CAST_NONE（截断/不舍入），`1` = CAST_RINT（四舍五入到最近偶数）。当前值 `0` 实际执行截断操作而非期望的 RINT 舍入，导致从高精度（float）Cast 回低精度（如 float16/int8）时精度损失超出预期，计算结果与 PyTorch 等框架的参考实现不一致。
- **触发条件**: 当使用 `LeakyReluCastDag` 模板（即输入类型 U 需要先 Cast 到 float 再计算的场景，如 U=half/int8），输出 Cast 回低精度时触发。特别是当 LeakyRelu 计算结果处于两个可表示值中间时，截断与四舍五入的差异最为明显。
- **测试方案**:
  1. 构造输入 tensor（float16 类型），使 LeakyRelu 输出值恰好位于两个 float16 可表示值的中间（如 0.1015625 附近）
  2. 设置 negative_slope 使负值部分产生需要舍入的结果
  3. 对比算子输出与 PyTorch `F.leaky_relu` 的 float16 输出，检查是否存在 1 ULP 的系统性偏差
  4. 验证修复：将 `CAST_MODE_RINT = 0` 改为 `CAST_MODE_RINT = 1` 后结果与参考一致

---

### Bug 2: LeakyReluDag 中标量参数类型与计算类型不匹配

- **位置**: 第 26 行 `using OpLeakRelu = Bind<Vec::LeakyRelu<U>, OpCopyInX, Placeholder::Var<T, 0>>;`
- **类型**: 类型不匹配
- **严重程度**: 中
- **描述**: 在 `LeakyReluDag` 模板中，向量计算使用类型 `U`（即 `Vec::LeakyRelu<U>`），但 negative_slope 标量参数使用类型 `T`（默认为 `float`），即 `Placeholder::Var<T, 0>`。当 `U != T`（例如 `U = half, T = float`）时，`LeakyRelu<half>` 接收一个 `float` 类型的标量参数，存在类型不匹配。根据 AscendC 接口规范，LeakyRelu 的标量参数类型应与计算 tensor 类型一致。这可能导致编译错误或隐式截断导致精度问题。
- **触发条件**: 当实例化 `LeakyReluDag<half, float>` 或其他 `U != T` 的情况时触发。若框架层做了隐式转换则不会编译报错，但可能有精度损失。
- **测试方案**:
  1. 以 `U=half, T=float` 实例化 `LeakyReluDag`，检查是否编译通过
  2. 若编译通过，设置 negative_slope 为无法精确用 half 表示的值（如 0.01），验证实际使用的标量值是否为完整 float 精度
  3. 对比 `LeakyReluDag<half>` 与 `LeakyReluCastDag<half>` 在相同输入下的输出差异
  4. 修复方案：将 `Placeholder::Var<T, 0>` 改为 `Placeholder::Var<U, 0>`，使标量类型与计算类型一致

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第37行 | 逻辑错误 | 高 | `CAST_MODE_RINT=0` 应为 `1`，当前实际执行截断而非四舍五入 |
| 2 | 第26行 | 类型不匹配 | 中 | `LeakyReluDag` 中计算类型为 `U` 但标量参数类型为 `T(float)`，U!=T 时不匹配 |
