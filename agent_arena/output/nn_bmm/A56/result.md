# Ascend NPU 算子代码审查报告

**文件**: `aclnn_batch_matmul.cpp`  
**审查日期**: 2026-08-03

---

### Bug 1: CreateBatchMatmulGraphImpl 空tensor分支逻辑被覆盖

- **位置**: 第 979-990 行, `CreateBatchMatmulGraphImpl` 函数
- **类型**: 逻辑错误 (控制流缺陷)
- **严重程度**: 严重 (Critical)
- **描述**: 当 `CheckBmmResIsEmpty` 返回 `true` 时，代码在第 985 行创建了 `BatchmmEmptyTensorGraph` 并赋值给 `matmulGraph`，但缺少 `return` 语句或 `else` 分支。程序继续执行到第 988 行，无条件地用 `BatchMatmulExecBmmOpGraph` 覆盖了 `matmulGraph`。导致空tensor场景永远不会走空tensor计算图路径。
- **触发条件**: 输入 self 或 mat2 的 batch/M/N 维度为 0 时，空tensor计算图被创建后立即被覆盖，实际仍走正常计算路径，可能导致非法内存访问或计算错误。
- **修复建议**: 在第 985 行 `matmulGraph = std::make_shared<BatchmmEmptyTensorGraph>(...)` 后添加 `return matmulGraph;`
- **测试方案**: 构造 shape 为 `[0, M, K]` 或 `[B, 0, K]` 或 mat2 shape `[B, K, 0]` 的输入 tensor，验证是否正确走入空tensor路径且不执行实际计算。

---

### Bug 2: ExecBmmOp 调用缺少参数 (编译错误)

- **位置**: 第 1051 行, `aclnnBatchMatMulWeightNzGetWorkspaceSize` 函数内调用
- **类型**: 接口调用错误 (参数缺失)
- **严重程度**: 严重 (Critical)
- **描述**: `ExecBmmOp` 函数签名 (第 919-920 行) 要求 6 个参数: `(self, mat2, out, cubeMathType, executor, isBaddbmm)`，但在第 1051 行的调用 `ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get())` 仅传入 5 个参数，缺少 `isBaddbmm` 参数。这将导致编译失败，或如果存在其他重载/默认值则可能产生未定义行为。
- **触发条件**: 编译时即报错；若存在隐式默认值，调用 WeightNz 路径时 isBaddbmm 状态不确定。
- **修复建议**: 补充第六个参数 `false`，改为 `ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get(), false)`
- **测试方案**: 编译验证；调用 `aclnnBatchMatMulWeightNzGetWorkspaceSize` 接口确认功能正常。

---

### Bug 3: CheckBmmResIsEmpty 假设输入为3D tensor

- **位置**: 第 104-108 行, `CheckBmmResIsEmpty` 函数
- **类型**: 边界条件错误 (维度假设)
- **严重程度**: 高 (High)
- **描述**: 函数直接使用 `FIRST_DIM=0`, `SECOND_DIM=1`, `THIRD_DIM=2` 访问 shape 维度，硬编码假设 self 为 `[B, M, K]`、mat2 为 `[B, K, N]` 的 3D tensor。但在 `CheckShape` 中，只有 mat2 和 out 被限制为 3D，self 可以是 2D~6D。当 self 为高维 tensor 时 (如 `[B1, B2, M, K]`)，`GetDim(FIRST_DIM)` 和 `GetDim(SECOND_DIM)` 不再是 batch 和 M，判断逻辑错误。
- **触发条件**: self 为 4D 及以上的 tensor，且其第 0 或第 1 维为 0，或实际 M 维不为 0 但第 1 维为 0 时。
- **修复建议**: 使用 `selfDimNum - 2` 和 `selfDimNum - 1` 获取 M 和 K 维度，batch 通过所有前置维度乘积计算。
- **测试方案**: 构造 4D 输入 `self=[2, 0, 4, 8]`，验证空tensor检测是否正确。

---

### Bug 4: ProcessEmptyTensor 硬编码3D维度索引

- **位置**: 第 253-254 行, `ProcessEmptyTensor` 函数
- **类型**: 边界条件错误 (维度假设)
- **严重程度**: 高 (High)
- **描述**: `bmmEmptyShape` 通过 `(self->GetViewShape())[0]`, `[1]`, `(mat2->GetViewShape())[2]` 构造输出shape，假设 self 和 mat2 均为 3D。若输入维度不为 3，生成的 output shape 将不正确，可能导致后续内存分配错误或数据越界。
- **触发条件**: 高维 tensor (>3D) 进入空tensor处理路径。
- **修复建议**: 使用 `GetDim(dimNum - 2)` 和 `GetDim(dimNum - 1)` 获取 M/N，batch 通过广播规则计算。
- **测试方案**: 构造 4D empty tensor 输入，验证输出 shape 是否正确。

---

### Bug 5: CheckShape 未校验 selfTensor 维度上限

- **位置**: 第 195-199 行, `CheckShape` 函数
- **类型**: 参数校验不完整
- **严重程度**: 中 (Medium)
- **描述**: 函数用 `OP_CHECK_WRONG_DIMENSION` 校验了 `otherTensor` 和 `outTensor` 为 3D，但对 `selfTensor` 没有维度上限校验。虽然 `GetBatchDim` 支持最多 6D，但在 `CheckBmmResIsEmpty`、`ProcessEmptyTensor` 等下游函数中假设 self 为 3D。维度校验不一致导致下游潜在越界。
- **触发条件**: self 为 >3D tensor 时通过 CheckShape 校验，但后续处理逻辑不兼容。
- **修复建议**: 对 selfTensor 增加维度范围校验（至少 2D，至多 6D），并确保与下游一致。
- **测试方案**: 传入 7D self tensor，验证是否正确报错拒绝。

---

### Bug 6: CheckBmmOp 中 CheckDtypeValid 被重复调用

- **位置**: 第 749-761 行, `CheckBmmOp` 函数
- **类型**: 冗余逻辑 / 潜在遗漏
- **严重程度**: 低 (Low)
- **描述**: 当 `mat2->GetStorageFormat() == FORMAT_FRACTAL_NZ` 时，先调用 `CheckDtypeValidWeightNz`，然后第 758 行无条件再次调用 `CheckDtypeValid`。对于 NZ 格式，这种双重校验逻辑冗余；更重要的是，`CheckDtypeValid` 的 dtype 支持列表包含 `DT_FLOAT`，而 `CheckDtypeValidWeightNz` 仅支持 FP16/BF16，可能导致 NZ 格式下 FP32 类型通过了第二次校验但实际不支持。
- **触发条件**: NZ 格式 + FP32 dtype 输入。
- **修复建议**: NZ 格式分支内不应再调用 `CheckDtypeValid`，应在 `else` 分支中调用或移除重复调用。
- **测试方案**: 传入 NZ 格式 FP32 tensor，验证是否正确拒绝。

---

### Bug 7: CheckSocIfBatchMatMulToMul91095 中 GetDim 返回 int64_t 赋值给 uint64_t

- **位置**: 第 549-554 行, `CheckSocIfBatchMatMulToMul91095` 函数
- **类型**: 类型安全隐患
- **严重程度**: 低 (Low)
- **描述**: `GetViewShape().GetDim()` 返回 `int64_t`，直接隐式转换为 `uint64_t`。若维度值为负数（某些框架用 -1 表示动态维度），转换为 uint64_t 会变成极大值，导致后续对齐计算和比较逻辑异常。
- **触发条件**: 输入 tensor 含有动态维度标记（-1）。
- **修复建议**: 增加维度值为正的断言，或显式检查后再转换。
- **测试方案**: 传入包含动态维度标记的 tensor shape，验证行为。

---

### Bug 8: Strides 交换索引使用常量 LAST_DIM 与 NUM_TWO 语义混淆

- **位置**: 第 878 行, `ExecBmmOpWithBias` 函数
- **类型**: 可维护性 / 潜在错误
- **严重程度**: 低 (Low)
- **描述**: `std::swap(strides[size - LAST_DIM], strides[size - NUM_TWO])` 中 `LAST_DIM=1` 作为"倒数第1"的偏移量，而 `NUM_TWO=2` 用作"倒数第2"的偏移量。但 `LAST_DIM` 的命名含义与 `PENULTIMATE_DIM=2` 矛盾（前者值为1代表"最后一维"，后者值为2代表"倒数第二维"）。此处应使用 `PENULTIMATE_DIM` 而非 `NUM_TWO` 以保持语义一致。当前数值正确但易引入后续维护 bug。
- **触发条件**: 不影响当前功能，但修改常量定义时可能引入错误。
- **修复建议**: 统一使用 `LAST_DIM` 和 `PENULTIMATE_DIM` 常量。
- **测试方案**: 代码审查确认。

---

## 汇总表

| 编号 | 位置 (行号) | 类型 | 严重程度 | 简述 |
|------|------------|------|----------|------|
| 1 | 979-990 | 逻辑错误 | Critical | 空tensor分支缺少return，被正常路径覆盖 |
| 2 | 1051 | 接口调用错误 | Critical | ExecBmmOp调用缺少isBaddbmm参数 |
| 3 | 104-108 | 边界条件错误 | High | CheckBmmResIsEmpty硬编码3D假设 |
| 4 | 253-254 | 边界条件错误 | High | ProcessEmptyTensor硬编码3D维度索引 |
| 5 | 195-199 | 参数校验不完整 | Medium | selfTensor缺少维度上限校验 |
| 6 | 749-761 | 冗余/遗漏逻辑 | Low | NZ格式下CheckDtypeValid重复调用 |
| 7 | 549-554 | 类型安全 | Low | int64_t到uint64_t隐式转换无防护 |
| 8 | 878 | 可维护性 | Low | Strides交换常量语义不一致 |
