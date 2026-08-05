# aclnn_matmul.cpp 代码审查报告

## Bug 列表

### Bug 1: K维度校验缺失（dimTensor2 == 1 或 2 时未比较K轴）

- **位置**: 第 180~192 行，`CheckShapeValid` 函数
- **类型**: 逻辑错误 / 输入校验缺失
- **严重程度**: 高
- **描述**: 当 `dimTensor2 == 1 || dimTensor2 == 2` 时，代码计算了 `selfKDim` 和 `mat2KDim`，但没有进行比较校验。K维不匹配的检查 `if (selfKDim != mat2KDim)` 仅在 `dimTensor2 >= 3` 的分支内执行。这意味着当 mat2 为 1D 或 2D 时，即使 self 的最后一维与 mat2 的第一维不匹配，也不会报错，直接进入后续计算流程，导致未定义行为或计算错误。
- **触发条件**: `self` shape 为 `[3, 5]`，`mat2` shape 为 `[4, 6]`（K轴分别为5和4，不匹配），且 mat2 维度为2。
- **测试方案**: 构造 K 维不匹配的 2D 输入对（如 self=[2,3], mat2=[4,5]），调用 `aclnnMatmulGetWorkspaceSize`，期望返回 `ACLNN_ERR_PARAM_INVALID`，实际会通过校验并在后续计算中产生错误。

---

### Bug 2: MatMulDimNumMatBGe3Graph 中 Unsqueeze 维度逻辑反转

- **位置**: 第 728~733 行，`MatMulDimNumMatBGe3Graph::PreProcess()`
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 当 `dimTensor1 == 1`（self 为 1D）时，代码使用 `dimData = {0}` 进行 unsqueeze，将 `[m]` 变为 `[1, m]`（2D）；当 `dimTensor1 == 2`（self 为 2D）时，使用 `dimData = {0, 1}`，将 `[n, m]` 变为 `[1, 1, n, m]`（4D）。逻辑应该反转：
  - 1D `[m]` 应 unsqueeze 在 `{0, 1}` 得到 `[1, 1, m]`（3D，添加 batch 维和行维）用于 batch matmul。
  - 2D `[n, m]` 应 unsqueeze 在 `{0}` 得到 `[1, n, m]`（3D，仅添加 batch 维）用于 batch matmul。
  
  当前实现中，1D 输入产生的 2D tensor 与 >=3D 的 mat2 做 batch matmul 时维度不足；2D 输入多添加了一个不必要的维度。
- **触发条件**: `self` 为 1D tensor（如 shape `[4]`），`mat2` 为 3D+ tensor（如 shape `[2, 4, 3]`），调用 matmul。
- **测试方案**: 构造 self=[4]（1D）, mat2=[2,4,3]（3D），执行 matmul，验证输出是否为预期的 [2,3] shape。当前实现会因维度不匹配导致 BatchMatmulProcess 失败或计算结果错误。

---

### Bug 3: const_cast 修改输入 tensor 的 ViewShape（副作用污染）

- **位置**: 第 460 行（`BuildMatMulWeightNzGraph`）和第 787~788 行（`MatMulWeightNzGraph::Impl`）
- **类型**: 数据完整性 / 副作用错误
- **严重程度**: 中
- **描述**: 代码通过 `const_cast<aclTensor*>(mat2)->SetViewShape(SwapLastTwoDimValue(...))` 直接修改了声明为 `const` 的输入 tensor 的 ViewShape。这违反了 const 语义契约，会永久修改调用者传入的 mat2 tensor 的状态。如果调用者后续再使用该 tensor（例如多次调用 matmul 或在其他运算中使用），其 shape 已被篡改，导致后续运算出错。
- **触发条件**: 使用 WeightNZ 格式且 mat2 存在转置属性（stride 表明转置）的情况下，多次调用 matmul 或在 matmul 后继续使用 mat2 tensor。
- **测试方案**: 构造转置的 NZ 格式 mat2 tensor，先调用一次 `aclnnMatmulWeightNzGetWorkspaceSize`，检查调用后 mat2 的 ViewShape 是否被修改。期望不变，实际已被 swap。

---

### Bug 4: CreateMatmulGraphImpl 中条件与注释互换

- **位置**: 第 839 行和第 842 行
- **类型**: 注释错误（可能引起维护误解）
- **严重程度**: 低
- **描述**: 第 839 行的代码条件为 `dimTensor1 >= 3 && (dimTensor2 == 1 || dimTensor2 == 2)`，但注释写的是 `// dimTensor1 is 1 or 2 && dimTensor2 >= 3`。第 842 行条件为 `(dimTensor1 == 1 || dimTensor1 == 2) && dimTensor2 >= 3`，注释写的是 `// dimTensor2 >= 3 && dimTensor1 is 1 or 2`（实际是对的但与839行注释互换了含义）。两行注释与实际代码逻辑相反，容易导致维护者误解代码意图。
- **触发条件**: 代码维护/阅读时产生歧义。
- **测试方案**: 代码审查确认即可，不影响运行时行为。

---

### Bug 5: 常量命名误导 NZ_K0_VALUE_32 实际值为 8

- **位置**: 第 53 行
- **类型**: 命名规范 / 可维护性
- **严重程度**: 低
- **描述**: `NZ_K0_VALUE_32 = 8`，常量名中的 "32" 指的是 FP32 数据类型（32-bit），而非 k0 的值。但与 `NZ_K0_VALUE_16 = 16` 对比时极易误解为 "k0 值为 32"。建议重命名为 `NZ_K0_VALUE_FP32 = 8` 和 `NZ_K0_VALUE_FP16 = 16` 以消除歧义。功能上数值正确（FP32 在 fractal NZ 格式中 k0=8），但命名造成混淆。
- **触发条件**: 后续开发者引用该常量时可能误解其含义。
- **测试方案**: 静态审查，确认 FP32 场景下 k0=8 是否为预期值（对 Ascend fractal NZ 格式规范应为8，正确）。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第 180~192 行 `CheckShapeValid` | 逻辑错误 | 高 | dimTensor2为1或2时K维度未校验，允许非法shape通过 |
| 2 | 第 728~733 行 `MatMulDimNumMatBGe3Graph::PreProcess` | 逻辑错误 | 高 | 1D和2D输入的unsqueeze维度逻辑反转，导致维度不匹配 |
| 3 | 第 460、787 行 | 副作用错误 | 中 | const_cast修改输入tensor的ViewShape，污染调用者状态 |
| 4 | 第 839、842 行 | 注释错误 | 低 | 条件分支注释与实际代码逻辑互换 |
| 5 | 第 53 行 | 命名规范 | 低 | NZ_K0_VALUE_32=8命名易引起误解 |
