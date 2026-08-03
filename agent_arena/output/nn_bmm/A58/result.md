# Ascend NPU 算子代码审查报告

**文件**: `aclnn_batch_matmul.cpp`  
**审查范围**: BatchMatMul 算子实现

---

### Bug 1: CreateBatchMatmulGraphImpl 缺少 else 分支导致空 Tensor 路径失效

**位置**: 第 980-991 行, `CreateBatchMatmulGraphImpl` 函数  
**类型**: 逻辑错误  
**严重程度**: 严重 (Critical)

**描述**:  
当 `CheckBmmResIsEmpty` 返回 true 时，代码创建了 `BatchmmEmptyTensorGraph` 赋值给 `matmulGraph`，但由于缺少 `else` 分支或 `return` 语句，紧接着无条件地用 `BatchMatmulExecBmmOpGraph` 覆盖了该变量。空 Tensor 分支永远不会生效。

```cpp
if (CheckBmmResIsEmpty(self, mat2)) {
    matmulGraph = std::make_shared<BatchmmEmptyTensorGraph>(...); // 被覆盖
}
// 缺少 else！
matmulGraph = std::make_shared<BatchMatmulExecBmmOpGraph>(...); // 总是执行
return matmulGraph;
```

**触发条件**: 当 self 的 batch=0、M=0 或 mat2 的 N=0 时，本应走空 Tensor 路径，但实际仍执行正常计算路径，可能导致非法内存访问或计算异常。

**测试方案**:  
构造 shape 为 [0, 3, 4] 和 [0, 4, 5] 的输入 tensor，调用 `aclnnBatchMatMulGetWorkspaceSize`，验证是否正确返回空 tensor 结果而非崩溃。

---

### Bug 2: ExecBmmOp 调用缺少 isBaddbmm 参数

**位置**: 第 1052 行, `aclnnBatchMatMulWeightNzGetWorkspaceSize` 函数  
**类型**: 参数缺失 / 编译错误  
**严重程度**: 严重 (Critical)

**描述**:  
`ExecBmmOp` 函数签名定义有 6 个参数（第 920-921 行），但在第 1052 行的调用中只传递了 5 个参数，缺少最后一个 `isBaddbmm` 布尔参数。

```cpp
// 定义 (line 920):
const aclTensor* ExecBmmOp(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* out,
    int8_t cubeMathType, aclOpExecutor* executor, bool isBaddbmm)

// 调用 (line 1052):
auto bmmOut = ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get());
// 缺少第6个参数 isBaddbmm
```

**触发条件**: 编译阶段即报错（除非头文件中声明了默认参数值）。若存在默认值隐式为 false，则功能正确但代码不一致。

**测试方案**:  
直接编译该文件，若无默认参数声明则编译失败；若存在默认值，需验证 WeightNz 路径的 isBaddbmm 语义是否正确。

---

### Bug 3: CheckBmmOp 中重复调用 CheckDtypeValid

**位置**: 第 754-759 行, `CheckBmmOp` 函数  
**类型**: 逻辑冗余 / 潜在逻辑错误  
**严重程度**: 中等 (Medium)

**描述**:  
当 mat2 为 `FORMAT_FRACTAL_NZ` 格式时，代码先调用 `CheckDtypeValidWeightNz`，然后无条件再次调用 `CheckDtypeValid`。WeightNz 校验要求 self/mat2/out dtype 完全一致，但 `CheckDtypeValid` 允许 dtype 不一致（会做 promotion），两者逻辑矛盾。对于非 NZ 格式，`CheckDtypeValid` 被调用了两次。

```cpp
if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
    CHECK_RET(CheckDtypeValidWeightNz(self, mat2, out), ...);
} else {
    CHECK_RET(CheckDtypeValid(self, mat2, bias, out, cubeMathType), ...);
}
CHECK_RET(CheckDtypeValid(self, mat2, bias, out, cubeMathType), ...); // 重复调用
```

**触发条件**: 所有经过 `CheckBmmOp` 的路径均受影响，NZ 格式下可能因重复校验逻辑不一致而产生误判。

**测试方案**:  
使用 FORMAT_FRACTAL_NZ 格式的 mat2，令 self 和 out 的 dtype 不同（如 fp16 和 fp32），观察是否通过 WeightNz 检查后被第二个 CheckDtypeValid 拦截。

---

### Bug 4: CheckShape 维度校验逻辑矛盾

**位置**: 第 198-214 行, `CheckShape` 函数  
**类型**: 逻辑冗余 / 死代码  
**严重程度**: 低 (Low)

**描述**:  
第 198-200 行使用 `OP_CHECK_WRONG_DIMENSION` 强制要求三个 tensor 维度恰好为 3（`SHAPE_LIMIT = 3`），但第 208 行又检查 `selfDimNum < 2 || otherDimNum < 2 || outDimNum < 2`。如果前面的检查生效（维度必须为3），则后续的 `< 2` 检查永远为 false，是死代码。若 `OP_CHECK_WRONG_DIMENSION` 的语义是"至少为3维"，则 `< 2` 的检查依然冗余。

**触发条件**: 无实际触发风险，但代码逻辑不清晰，维护时容易引入错误。

**测试方案**:  
传入 2D tensor（如 shape [3,4]）验证 `OP_CHECK_WRONG_DIMENSION` 是否拦截，确认校验逻辑的实际语义。

---

### Bug 5: const 指针上调用非 const 方法 SetStorageShape

**位置**: 第 897 行, `ExecBmmOpWithBias` 函数  
**类型**: const 正确性违规  
**严重程度**: 中等 (Medium)

**描述**:  
`contiguousMat2` 的类型为 `const aclTensor*`（从 `reformatMat2` 推导而来，第 863 行），但第 897 行直接调用了 `contiguousMat2->SetStorageShape(...)`，这是对 const 对象调用修改方法。

```cpp
auto contiguousMat2 = reformatMat2; // const aclTensor*
// ...
contiguousMat2->SetStorageShape(mat2->GetStorageShape()); // const violation
```

**触发条件**: mat2 为 FORMAT_FRACTAL_NZ 格式时执行到该行。若框架未将 SetStorageShape 标记为 const（正常不应是 const），则编译报错；若通过 mutable 实现则有隐式副作用风险。

**测试方案**:  
使用严格 const 正确性的编译选项编译，验证是否报错。在运行时使用 NZ 格式 mat2 验证 storage shape 是否被正确设置。

---

### Bug 6: CheckBmmResIsEmpty 未检查 K 维度为 0 的情况

**位置**: 第 104-108 行, `CheckBmmResIsEmpty` 函数  
**类型**: 边界条件遗漏  
**严重程度**: 中等 (Medium)

**描述**:  
函数检查 B=0、M=0、N=0 的空 tensor 情况，但未检查 K=0（即 `self->GetViewShape().GetDim(THIRD_DIM) == 0`）。当 K=0 时，矩阵乘法结果应全为 0，但如果不走空 tensor 路径，后续 BMM 内核可能因 K=0 产生未定义行为。

```cpp
// [B,M,K] @ [B,K,N] = [B,M,N]
static inline bool CheckBmmResIsEmpty(const aclTensor* self, const aclTensor* mat2)
{
    return self->GetViewShape().GetDim(FIRST_DIM) == 0 ||    // B==0
           self->GetViewShape().GetDim(SECOND_DIM) == 0 ||   // M==0
           mat2->GetViewShape().GetDim(THIRD_DIM) == 0;      // N==0
    // 缺少: self->GetViewShape().GetDim(THIRD_DIM) == 0     // K==0
}
```

**触发条件**: 输入 self 的 shape 为 [B, M, 0]，mat2 的 shape 为 [B, 0, N]（K 维度为 0）。

**测试方案**:  
构造 shape [2, 3, 0] 和 [2, 0, 4] 的输入，验证是否正确返回全零的 [2, 3, 4] 结果而非异常。

---

### Bug 7: ProcessEmptyTensor 输出 shape 未考虑 batch 广播

**位置**: 第 252-268 行, `ProcessEmptyTensor` 函数  
**类型**: 逻辑错误  
**严重程度**: 低 (Low)

**描述**:  
空 tensor 处理时使用 `self->GetViewShape()[0]` 作为 batch 维度，但 BMM 支持 batch 广播（self batch=1, mat2 batch=N 时输出 batch=N）。此处未取 max(self_batch, mat2_batch)。

```cpp
op::Shape bmmEmptyShape = {(self->GetViewShape())[0], (self->GetViewShape())[1], (mat2->GetViewShape())[2]};
// 应为 max(self[0], mat2[0]) 或按广播规则计算
```

**触发条件**: self shape [1, 0, K]，mat2 shape [8, K, 0] 时，输出 batch 应为 8 但实际为 1。

**测试方案**:  
构造 batch 不等的空 tensor（如 [1,0,4] 和 [8,4,0]），验证输出 shape 是否为 [8,0,0]。

---

### Bug 8: CheckTransNonContiguousShapeSupport 中 adjX2 参数缺失

**位置**: 第 763-817 行, `CheckTransNonContiguousShapeSupport` 函数  
**类型**: 参数遗漏  
**严重程度**: 低 (Low)

**描述**:  
函数签名不包含 `adjX2` 参数，但内部假设 mat2 的 N 维度在最后一维（第 783 行）。当 mat2 存在转置时，N 和 K 的位置会交换，但此处未考虑。注释也说"非连续场景 viewshape 一定是 bkn 格式"，但这一假设未在代码中显式校验。

**触发条件**: mat2 具有非标准内存布局且逻辑上需要转置时，维度判断可能错误。

**测试方案**:  
构造需要 adjX2=true 的非连续 mat2 tensor，验证 shape 检查是否正确。

---

## 汇总表

| # | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|---|------|----------|----------|----------|
| 1 | L980-991 | 逻辑错误 | Critical | CreateBatchMatmulGraphImpl 缺少 else，空 Tensor 路径死代码 |
| 2 | L1052 | 参数缺失 | Critical | ExecBmmOp 调用缺少 isBaddbmm 参数 |
| 3 | L754-759 | 逻辑冗余 | Medium | CheckBmmOp 重复调用 CheckDtypeValid，NZ 格式逻辑矛盾 |
| 4 | L198-214 | 死代码 | Low | CheckShape 维度检查逻辑矛盾 |
| 5 | L897 | const 违规 | Medium | const 指针调用 SetStorageShape 修改方法 |
| 6 | L104-108 | 边界遗漏 | Medium | CheckBmmResIsEmpty 未检查 K==0 |
| 7 | L252-255 | 逻辑错误 | Low | ProcessEmptyTensor 未考虑 batch 广播 |
| 8 | L763-783 | 参数遗漏 | Low | CheckTransNonContiguousShapeSupport 缺少 adjX2 考虑 |
