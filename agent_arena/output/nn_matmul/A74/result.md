# Ascend NPU 算子代码审查报告 - aclnn_matmul.cpp (A74)

## Bug 列表

### Bug 1: aclnnMatmulWeightNzGetWorkspaceSize 中资源泄漏

- **位置**: 第 929-934 行
- **类型**: 资源泄漏 (Resource Leak)
- **严重程度**: 高
- **描述**: 在 `aclnnMatmulWeightNzGetWorkspaceSize` 函数中，`CREATE_EXECUTOR()` 在参数检查 `CheckWeightNzParam` 之前调用。若参数检查失败，`CHECK_RET` 直接返回错误码，但 `uniqueExecutor` 已创建却未被释放，导致内存泄漏。对比 `aclnnMatmulGetWorkspaceSize`（第 886 行），其正确地将参数检查放在 executor 创建之前。
- **触发条件**: 调用 `aclnnMatmulWeightNzGetWorkspaceSize` 时传入不合法参数（如不支持的数据类型、不匹配的 shape、不支持的 SOC 版本等），使 `CheckWeightNzParam` 返回失败。
- **测试方案**: 传入 dtype 为 DT_FLOAT 的 mat2（NZ 格式不支持 FP32），验证函数返回错误码后，通过内存检测工具（如 ASan/Valgrind）检测是否存在 executor 对象的内存泄漏。

---

### Bug 2: CheckShapeValid 中缺少对 self 的最大维度检查

- **位置**: 第 171 行
- **类型**: 逻辑缺陷 (Missing Validation)
- **严重程度**: 中
- **描述**: `CheckShapeValid` 函数中仅对 `mat2` 调用了 `OP_CHECK_MAX_DIM` 检查最大维度（6维），未对 `self` 进行同样的检查。若 `self` 的维度超过 `MAX_SUPPORT_MATMUL_DIMS_NUMS`（6），将绕过维度限制进入后续计算逻辑，可能导致未定义行为或计算错误。
- **触发条件**: 传入 `self` 维度数大于 6（例如 7 维张量），同时 `mat2` 维度数 <= 6。
- **测试方案**: 构造一个 7 维的 `self` 张量和合法的 2 维 `mat2` 张量调用 matmul，验证是否能正确返回 `ACLNN_ERR_PARAM_INVALID` 错误。

---

### Bug 3: 常量命名错误 NZ_K0_VALUE_32

- **位置**: 第 53 行
- **类型**: 命名错误 (Naming Bug / 可维护性缺陷)
- **严重程度**: 低
- **描述**: `NZ_K0_VALUE_32` 的实际值为 `8`，而非 `32`。该变量用于 FP32 数据类型的 NZ 格式 K0 值（因 FP32 占 4 字节，一个 block 能容纳 8 个元素）。变量名中的 "32" 极易被误解为值是 32，增加了后续维护和使用时出错的风险。应命名为 `NZ_K0_VALUE_FP32` 或 `NZ_K0_VALUE_8`。
- **触发条件**: 开发者在其他位置引用该常量时，误以为其值为 32 而非 8，导致逻辑错误。
- **测试方案**: 代码审查确认所有引用 `NZ_K0_VALUE_32` 的位置是否正确理解了其语义；静态分析工具检查命名一致性。

---

### Bug 4: MatMulDimNumMatBGe3Graph 中 Unsqueeze 维度逻辑不完备

- **位置**: 第 733-738 行
- **类型**: 逻辑缺陷 (Logic Error)
- **严重程度**: 中
- **描述**: 当 `dimTensor1 == 2` 时，代码固定对 `matA` 在 `{0, 1}` 两个维度做 Unsqueeze，使 2D 张量 `[m, k]` 变为 `[1, 1, m, k]`（4D）。但此时 `mat2` 可能是 3D `[B, k, n]`，BatchMatmul 会收到 4D × 3D 的输入。正确做法应根据 `mat2` 的实际 batch 维数动态决定 Unsqueeze 的维数（3D mat2 只需 Unsqueeze 一个 batch 维 `{0}`，使 matA 成为 `[1, m, k]`）。当前的固定逻辑在 mat2 为 3D 时会引入多余的维度，可能导致输出 shape 不符合预期。
- **触发条件**: `self` 为 2D 张量（如 `[m, k]`），`mat2` 为 3D 张量（如 `[B, k, n]`），调用 matmul。
- **测试方案**: 构造 `self=[3, 4]`（2D）和 `mat2=[2, 4, 5]`（3D）进行 matmul，验证输出 shape 是否正确为 `[2, 3, 5]`，并检查结果数值的正确性。

---

### Bug 5: CheckWeightNzInputParams 函数已定义但未被调用

- **位置**: 第 359-379 行
- **类型**: 代码一致性缺陷 (Dead Code / Missing Integration)
- **严重程度**: 中
- **描述**: `CheckWeightNzInputParams`（第 359 行）实现了基于 `socRule->CheckInput` 的新校验逻辑（与 `CheckInputParams` 一致），但在 `aclnnMatmulWeightNzGetWorkspaceSize`（第 933 行）中仍使用旧的 `CheckWeightNzParam`。这意味着 WeightNz 路径未采用与普通 Matmul 路径相同的新版 SOC 规则校验，可能遗漏新增的校验规则。
- **触发条件**: 当新 SOC 规则中添加了 `CheckWeightNzParam` 未覆盖的校验逻辑时，WeightNz 路径会放过不合法的输入。
- **测试方案**: 对比 `CheckWeightNzInputParams` 和 `CheckWeightNzParam` 的校验覆盖范围，使用不满足新 socRule 但满足旧逻辑的参数调用 WeightNz 接口，验证是否被正确拦截。

---

### Bug 6: 注释与代码逻辑不匹配

- **位置**: 第 179 行、第 844 行
- **类型**: 注释错误 (Comment Mismatch)
- **严重程度**: 低
- **描述**:
  - 第 179 行注释 `"tensor1 dims number is 1 OR tensor2 dims number is 2"` 但实际条件为 `dimTensor2 == 1 || dimTensor2 == 2`，仅涉及 tensor2。
  - 第 844 行注释 `"dimTensor1 is 1 or 2 && dimTensor2 >= 3"` 但实际条件为 `dimTensor1 >= 3 && (dimTensor2 == 1 || dimTensor2 == 2)`，与注释完全相反。
- **触发条件**: 开发者依据错误注释理解代码逻辑，做出错误修改。
- **测试方案**: 代码审查确认注释与实际逻辑一致性。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 929-934 行 | 资源泄漏 | 高 | executor 创建后参数检查失败未释放，内存泄漏 |
| 2 | 第 171 行 | 逻辑缺陷 | 中 | 仅检查 mat2 最大维度，遗漏 self 的维度上限校验 |
| 3 | 第 53 行 | 命名错误 | 低 | NZ_K0_VALUE_32 实际值为 8，名称误导 |
| 4 | 第 733-738 行 | 逻辑缺陷 | 中 | dimTensor1==2 时固定 Unsqueeze 两维，未适配 mat2 维度 |
| 5 | 第 359-379 行 | 代码一致性 | 中 | CheckWeightNzInputParams 已定义未调用，新校验规则未生效 |
| 6 | 第 179, 844 行 | 注释错误 | 低 | 注释与实际条件逻辑不匹配 |
