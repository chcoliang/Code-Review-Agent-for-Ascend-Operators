# Ascend NPU BatchMatMul 算子代码审查报告

## Bug 列表

### Bug 1: CreateBatchMatmulGraphImpl 缺少 return 语句导致空 tensor 路径失效

- **位置**: 第 975-986 行, `CreateBatchMatmulGraphImpl` 函数
- **类型**: 逻辑错误 / 控制流缺陷
- **严重程度**: 严重 (Critical)
- **描述**: 当 `CheckBmmResIsEmpty` 返回 true 时，函数创建了 `BatchmmEmptyTensorGraph` 并赋值给 `matmulGraph`，但没有 `return` 语句。程序继续执行到下一行，`matmulGraph` 被 `BatchMatmulExecBmmOpGraph` 覆盖，导致空 tensor 场景永远不会走入 `BatchmmEmptyTensorGraph` 计算图。
- **触发条件**: 输入 self 或 mat2 的 batch/M/N 维度为 0（空 tensor），通过 `aclnnBatchMatMulGetWorkspaceSize` 进入（虽然上层有提前返回逻辑，但 `CreateBatchMatmulGraphImpl` 在 line 1011 仍被调用的场景实际上不应到达此处——因为 line 1004 已提前返回。但函数本身逻辑有明显缺陷，若未来重构移除上层提前返回则会触发）。
- **测试方案**: 移除 `aclnnBatchMatMulGetWorkspaceSize` 中 line 1003-1008 的空 tensor 提前返回，传入 shape 为 [0, M, K] 的 self tensor，验证是否走入 `BatchmmEmptyTensorGraph`。

```cpp
// 当前代码（错误）:
if (CheckBmmResIsEmpty(self, mat2)) {
    matmulGraph = std::make_shared<BatchmmEmptyTensorGraph>(...);
}
// 缺少 return matmulGraph;
matmulGraph = std::make_shared<BatchMatmulExecBmmOpGraph>(...);
return matmulGraph;

// 修复:
if (CheckBmmResIsEmpty(self, mat2)) {
    matmulGraph = std::make_shared<BatchmmEmptyTensorGraph>(...);
    return matmulGraph;  // 添加 return
}
matmulGraph = std::make_shared<BatchMatmulExecBmmOpGraph>(...);
return matmulGraph;
```

---

### Bug 2: ExecBmmOp 调用缺少 isBaddbmm 参数

- **位置**: 第 1047 行, `aclnnBatchMatMulWeightNzGetWorkspaceSize` 函数
- **类型**: 参数缺失 / 编译错误
- **严重程度**: 严重 (Critical)
- **描述**: `ExecBmmOp` 函数签名（第 915-916 行）需要 6 个参数：`self, mat2, out, cubeMathType, executor, isBaddbmm`。但第 1047 行调用仅传了 5 个参数，缺少最后的 `isBaddbmm` 布尔参数。如果编译器没有提供默认值，则会导致编译失败；如果通过隐式转换或其他机制编译通过，则行为未定义。
- **触发条件**: 编译阶段即应报错。若通过某种隐式机制通过编译，则所有 `aclnnBatchMatMulWeightNz` 路径都会受影响。
- **测试方案**: 直接编译该文件，检查编译错误；或在 WeightNz 路径中调用 `aclnnBatchMatMulWeightNzGetWorkspaceSize` 验证。

```cpp
// 当前代码（错误）:
auto bmmOut = ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get());

// 修复:
auto bmmOut = ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get(), false);
```

---

### Bug 3: CheckBmmOp 中 CheckDtypeValid 重复调用

- **位置**: 第 749-754 行, `CheckBmmOp` 函数
- **类型**: 逻辑冗余 / 潜在错误
- **严重程度**: 中等 (Medium)
- **描述**: 当 `mat2` 的 StorageFormat 为 `FORMAT_FRACTAL_NZ` 时，代码先调用 `CheckDtypeValidWeightNz`（line 750），然后无条件再调用 `CheckDtypeValid`（line 754）。这意味着 NZ 格式的 tensor 会同时受到两种不同 dtype 校验规则的约束，可能导致本应合法的 NZ 场景被错误拒绝（因为 WeightNZ 仅支持 fp16/bf16，而通用检查可能有不同的类型提升逻辑）。正确的写法应是 `else` 互斥分支，或第 754 行应放在 else 分支内。
- **触发条件**: mat2 为 FRACTAL_NZ 格式，且 dtype 满足 WeightNz 规则但不满足通用规则（例如在不支持 bf16 的平台上传入 bf16 + NZ 格式）。
- **测试方案**: 在不支持 bf16 的 SoC 上，传入 FRACTAL_NZ 格式的 bf16 mat2 tensor，观察是否触发误报错。

```cpp
// 当前代码（错误）:
if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
    CHECK_RET(CheckDtypeValidWeightNz(self, mat2, out), ACLNN_ERR_PARAM_INVALID);
} else {
    CHECK_RET(CheckDtypeValid(self, mat2, bias, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);
}
CHECK_RET(CheckDtypeValid(self, mat2, bias, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);  // 冗余/错误

// 修复: 删除第 754 行的重复调用
```

---

### Bug 4: ExecBmmOpWithBias 中 Contiguous/CreateView 使用原始 self 而非 reformatSelf

- **位置**: 第 859-863 行, `ExecBmmOpWithBias` 函数
- **类型**: 逻辑错误 / 数据格式不一致
- **严重程度**: 中等 (Medium)
- **描述**: 第 828-829 行将 `self` 通过 `l0op::ReFormat` 转换为 ND 格式得到 `reformatSelf`。但在第 859-863 行的 contiguous 处理中，`IsTransposeLastTwoDims` 和 `CreateView`/`Contiguous` 仍然使用原始的 `self` 而非 `reformatSelf`。这跳过了 ReFormat 步骤，可能导致后续计算使用了未转换格式的 tensor。
- **触发条件**: self 的原始格式非 ND 且非 FRACTAL_NZ（如 NHWC 等），同时 self 满足转置条件。
- **测试方案**: 传入 FORMAT_NCHW 格式的 self tensor，且其 stride 满足最后两维转置条件，检查是否正确进行格式转换。

```cpp
// 当前代码（错误）:
transposeSelf = Ops::NN::IsTransposeLastTwoDims(self);  // 应使用 reformatSelf
if (transposeSelf) {
    contiguousSelf = executor->CreateView(self, ...);   // 应使用 reformatSelf
} else {
    contiguousSelf = l0op::Contiguous(self, executor);  // 应使用 reformatSelf
}

// 修复: 将 self 替换为 reformatSelf
```

---

### Bug 5: const aclTensor* 上调用非 const 方法 SetStorageShape

- **位置**: 第 893 行, `ExecBmmOpWithBias` 函数
- **类型**: const 正确性违反 / 潜在编译错误
- **严重程度**: 低 (Low)
- **描述**: `contiguousMat2` 的类型为 `const aclTensor*`（从第 857 行 `auto contiguousMat2 = reformatMat2` 推导，reformatMat2 类型为 `const aclTensor*`）。第 893 行直接在 `contiguousMat2` 上调用 `SetStorageShape`（非 const 成员函数），违反 const 正确性。代码中其他类似场景（如第 875 行）使用了 `const_cast`。
- **触发条件**: mat2 格式为 FORMAT_FRACTAL_NZ 时触发该代码路径。
- **测试方案**: 使用严格的编译器设置编译此文件（-Werror），传入 FRACTAL_NZ 格式的 mat2。

```cpp
// 修复:
const_cast<aclTensor*>(contiguousMat2)->SetStorageShape(mat2->GetStorageShape());
```

---

### Bug 6: CheckBmmResIsEmpty 硬编码 3D 维度索引，但调用者不保证 3D

- **位置**: 第 104-108 行, `CheckBmmResIsEmpty` 函数; 调用点第 819 行
- **类型**: 越界访问风险
- **严重程度**: 中等 (Medium)
- **描述**: `CheckBmmResIsEmpty` 使用 `FIRST_DIM(0)`, `SECOND_DIM(1)`, `THIRD_DIM(2)` 硬编码访问 shape 维度，假设输入是 3D tensor。但在 `ExecBmmOpWithBias`（第 819 行）调用时，输入 tensor 可能有更高维度（4D-6D，因为 BMM 支持多 batch 维度）。此时 `GetDim(THIRD_DIM)` 取到的不是 mat2 的 N 维度，而是中间的 batch 维度，导致空 tensor 判断不正确。
- **触发条件**: 传入 4D+ shape 的 tensor（如 [2, 3, 4, 0]），`GetDim(2)` 返回 4 而非 0，空 tensor 判断漏报。
- **测试方案**: 传入 shape 为 [2, 3, 4, 0] 的 mat2（N=0），验证是否被正确识别为空 tensor。

```cpp
// 修复: 使用末尾偏移
static inline bool CheckBmmResIsEmpty(const aclTensor* self, const aclTensor* mat2)
{
    auto selfDim = self->GetViewShape().GetDimNum();
    auto mat2Dim = mat2->GetViewShape().GetDimNum();
    return self->GetViewShape().GetDim(0) == 0 ||
           self->GetViewShape().GetDim(selfDim - 2) == 0 ||
           mat2->GetViewShape().GetDim(mat2Dim - 1) == 0;
}
```

---

## 汇总表

| 编号 | 位置 (行号) | Bug 类型 | 严重程度 | 简述 |
|------|------------|----------|----------|------|
| 1 | 975-986 | 逻辑错误/控制流 | 严重 | `CreateBatchMatmulGraphImpl` 空 tensor 分支缺少 return，空 tensor 计算图被覆盖 |
| 2 | 1047 | 参数缺失 | 严重 | `ExecBmmOp` 调用缺少第 6 个参数 `isBaddbmm` |
| 3 | 749-754 | 逻辑冗余 | 中等 | `CheckBmmOp` 中 `CheckDtypeValid` 在 NZ 分支后被无条件重复调用 |
| 4 | 859-863 | 逻辑错误 | 中等 | Contiguous 处理使用原始 `self` 而非已 ReFormat 的 `reformatSelf` |
| 5 | 893 | const 正确性 | 低 | 在 `const aclTensor*` 上调用非 const 方法 `SetStorageShape` |
| 6 | 104-108 | 越界/逻辑错误 | 中等 | `CheckBmmResIsEmpty` 硬编码 3D 索引，高维 tensor 场景判断错误 |
