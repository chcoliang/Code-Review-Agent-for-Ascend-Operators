# Ascend NPU 算子代码审查报告 - aclnn_matmul.cpp (A66)

## Bug 列表

### Bug 1: DTYPE_SUPPORT_LIST 缺少 DT_BF16 数据类型

- **位置**: 第 56-59 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `DTYPE_SUPPORT_LIST` 和 `DTYPE_SUPPORT_LIST_WITHOUT_BF16` 两个列表内容完全相同，都只包含 `DT_FLOAT` 和 `DT_FLOAT16`。`DTYPE_SUPPORT_LIST` 应该额外包含 `DT_BF16`，但实际未包含。这导致在第 114 行的三元选择 `bf16flag ? DTYPE_SUPPORT_LIST : DTYPE_SUPPORT_LIST_WITHOUT_BF16` 毫无意义——即使 SoC 支持 BF16，`OP_CHECK_DTYPE_NOT_SUPPORT` 仍然会拒绝 BF16 类型的输入。
- **触发条件**: 在 Ascend910B~910E 平台上，输入 tensor 的 dtype 为 `DT_BF16` 时，`CheckDtypeValid` 会错误地拒绝合法的 BF16 输入。
- **测试方案**: 在 Ascend910B 平台上，构造 self 和 mat2 dtype 均为 BF16 的 matmul 用例，验证是否能正常执行（通过旧 `CheckWeightNzParam` 路径触发时会失败）。

---

### Bug 2: MatMulDimNumMatBGe3Graph 中 dimTensor1==2 时 Unsqueeze 维度错误

- **位置**: 第 737-738 行 (`MatMulDimNumMatBGe3Graph::PreProcess`)
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 当 `dimTensor1 == 2` 时，代码设置 `dimData = FVector<int64_t>{0, 1}`，对 2D 的 matA（shape [M, K]）在第 0 维和第 1 维各做一次 unsqueeze，结果变为 [1, 1, M, K]（4D）。然而 matB 为 3D（如 [B, K, N]），后续 `BatchMatmulProcess` 会处理 4D vs 3D 的 batch 维度不匹配。正确做法应仅 unsqueeze 一次（`dimData = {0}`），使 matA 变为 [1, M, K]（3D），以便与 matB 正确广播。
- **触发条件**: self 为 2D tensor（如 shape [4, 8]），mat2 为 3D tensor（如 shape [2, 8, 6]），调用 `aclnnMatmul` 走新的 Graph 路径。
- **测试方案**: 构造 self=[4,8], mat2=[2,8,6] 的 matmul 测试用例，验证输出 shape 和数值是否与 PyTorch `torch.matmul` 一致。

---

### Bug 3: 常量 NZ_K0_VALUE_32 命名与实际值不一致

- **位置**: 第 53 行
- **类型**: 命名错误 / 潜在逻辑隐患
- **严重程度**: 中
- **描述**: 常量 `NZ_K0_VALUE_32` 的值为 8，而非名称暗示的 32。该常量用于 float32 类型时的 NZ 格式 block 大小计算（第 392 行）。虽然 float32 占 4 字节、16 元素/block ÷ (4/2) = 8 的逻辑可能成立，但命名中的 "32" 极易误导开发者。若该值语义为 "float32 对应的 K0"，应命名为 `NZ_K0_VALUE_FP32` 或类似。如果命名意图是"32字节对齐对应的元素数"也仍有歧义。
- **触发条件**: 后续维护者误以为该值为 32 并据此修改相关逻辑时可能引入 bug。当 mat2 为 float32 的 NZ 格式时，若值本应为其他数字则会导致 shape 计算错误。
- **测试方案**: 构造 float32 类型的 NZ 格式 weight tensor，验证 `GetWeightNzShape` 计算出的 storage shape 与实际 NZ 格式规范是否一致。

---

### Bug 4: const_cast 修改 const 指针指向的 tensor 对象

- **位置**: 第 466 行, 第 794 行
- **类型**: 未定义行为 / 接口违约
- **严重程度**: 中
- **描述**: 代码通过 `const_cast<aclTensor*>(mat2)->SetViewShape(...)` 修改了以 `const aclTensor*` 传入的 mat2 参数的 viewShape。这违反了 const 契约：调用者传入 const 指针意味着不期望对象被修改。在 `BuildMatMulWeightNzGraph`（第 466 行）中，修改后的 mat2 的 viewShape 会影响后续对同一 tensor 的所有使用。在 `MatMulWeightNzGraph::Impl`（第 794 行）中同样存在此问题。
- **触发条件**: mat2 为转置的 NZ 格式 tensor 时，`GetTransposeAttrValue` 返回 true，触发 `SetViewShape`。若调用者在 `GetWorkspaceSize` 后继续使用 mat2 tensor，其 viewShape 已被篡改。
- **测试方案**: 创建一个转置的 NZ 格式 mat2，调用 `aclnnMatmulWeightNzGetWorkspaceSize` 后，检查 mat2 的 viewShape 是否被意外修改。

---

### Bug 5: aclnnMatmulWeightNzGetWorkspaceSize 使用旧版参数校验函数

- **位置**: 第 934 行
- **类型**: 逻辑不一致 / 功能缺陷
- **严重程度**: 中
- **描述**: `aclnnMatmulWeightNzGetWorkspaceSize` 调用的是旧版 `CheckWeightNzParam`（内部使用 `CheckDtypeValid`，依赖有 bug 的 `DTYPE_SUPPORT_LIST`），而非新版 `CheckWeightNzInputParams`（内部使用 `BuildRule()` 进行 SoC 规则检查）。对比 `aclnnMatmulGetWorkspaceSize`（第 887 行）已经使用新版 `CheckInputParams`，WeightNz 路径未同步更新。
- **触发条件**: 任何通过 `aclnnMatmulWeightNzGetWorkspaceSize` 入口的调用，都使用旧的、不完善的参数校验逻辑。
- **测试方案**: 构造一个新版 `BuildRule()` 能检测到但旧版 `CheckDtypeValid` 无法检测到的非法输入组合，验证 WeightNz 路径是否能正确拦截。

---

### Bug 6: CheckShapeValid 对 dimTensor1==1 且 dimTensor2==2 时的 K 轴校验逻辑注释与条件不匹配

- **位置**: 第 180 行
- **类型**: 代码质量 / 潜在维护风险
- **严重程度**: 低
- **描述**: 第 180 行的条件 `else if (dimTensor2 == 1 || dimTensor2 == 2)` 的注释写的是 "tensor1 dims number is 1 OR tensor2 dims number is 2"，但实际条件仅检查了 `dimTensor2`。当 `dimTensor1 >= 2` 且 `dimTensor2 == 1` 时（如 [M,K] @ [K]），K 轴校验使用 `mat2Shape.GetDim(0)` 是正确的。但注释的误导性可能导致后续维护者错误理解分支逻辑。
- **触发条件**: 不会直接导致运行时错误，但增加了维护风险。
- **测试方案**: 代码审查确认所有 dimTensor1/dimTensor2 组合下的 K 轴校验逻辑正确性。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 56-59 行 | 逻辑错误 | 高 | DTYPE_SUPPORT_LIST 缺少 DT_BF16，BF16 输入永远被旧路径拒绝 |
| 2 | 第 737-738 行 | 逻辑错误 | 高 | dimTensor1==2 时多做了一次 unsqueeze，导致 batch 维度不匹配 |
| 3 | 第 53 行 | 命名错误 | 中 | NZ_K0_VALUE_32 实际值为 8，命名严重误导 |
| 4 | 第 466, 794 行 | 未定义行为 | 中 | const_cast 修改 const 参数，违反接口契约 |
| 5 | 第 934 行 | 逻辑不一致 | 中 | WeightNz 路径使用旧版校验函数，未与主路径同步更新 |
| 6 | 第 180 行 | 代码质量 | 低 | 注释与条件不匹配，增加维护风险 |
