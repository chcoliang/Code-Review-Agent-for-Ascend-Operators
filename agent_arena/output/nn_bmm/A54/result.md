# Ascend NPU 算子代码审查报告

**文件**: `aclnn_batch_matmul.cpp`

---

### Bug 1: CreateBatchMatmulGraphImpl 空tensor分支结果被无条件覆盖

- **位置**: 第 980-991 行, `CreateBatchMatmulGraphImpl` 函数
- **类型**: 逻辑错误 (控制流缺陷)
- **严重程度**: 严重 (Critical)
- **描述**: 当 `CheckBmmResIsEmpty(self, mat2)` 为 true 时，创建了 `BatchmmEmptyTensorGraph` 并赋值给 `matmulGraph`，但紧接着在没有 `else` 保护的情况下，又无条件创建了 `BatchMatmulExecBmmOpGraph` 覆盖了前一个结果。导致空 tensor 场景永远不会走入 `BatchmmEmptyTensorGraph` 路径，空 tensor 的零填充逻辑失效，可能对空 tensor 执行非法计算。
- **触发条件**: 输入 self 的 batch=0 或 M=0，或 mat2 的 N=0 时触发。
- **测试方案**: 构造 shape 为 [0, 4, 4]、[2, 0, 4]、[2, 4, 0] 的输入 tensor 调用 `aclnnBatchMatMulGetWorkspaceSize`，验证是否正确返回空 tensor 而非崩溃。

```cpp
// 缺陷代码 (第985-989行):
if (CheckBmmResIsEmpty(self, mat2)) {
    matmulGraph = std::make_shared<BatchmmEmptyTensorGraph>(...);
}
// 缺少 else！下面的赋值会覆盖上面的结果
matmulGraph = std::make_shared<BatchMatmulExecBmmOpGraph>(...);

// 修复: 添加 else 分支
if (CheckBmmResIsEmpty(self, mat2)) {
    matmulGraph = std::make_shared<BatchmmEmptyTensorGraph>(...);
} else {
    matmulGraph = std::make_shared<BatchMatmulExecBmmOpGraph>(...);
}
```

---

### Bug 2: ExecBmmOp 调用缺少 isBaddbmm 参数

- **位置**: 第 1045 行, `aclnnBatchMatMulWeightNzGetWorkspaceSize` 函数内
- **类型**: 参数缺失 (编译错误/未定义行为)
- **严重程度**: 严重 (Critical)
- **描述**: `ExecBmmOp` 函数签名(第920-921行)定义了6个参数，最后一个为 `bool isBaddbmm`。但在第1045行调用时只传递了5个参数，缺少 `isBaddbmm` 参数。如果编译器没有为该参数提供默认值（代码中未见默认值声明），则会导致编译失败；若存在隐式默认值或函数重载，则可能导致未定义行为。
- **触发条件**: 编译 `aclnnBatchMatMulWeightNz` 相关路径时触发。
- **测试方案**: 编译该文件验证是否报错；若编译通过，则用 WeightNz 场景调用验证 isBaddbmm 逻辑是否正确。

```cpp
// 缺陷代码 (第1045行):
auto bmmOut = ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get());

// 修复:
auto bmmOut = ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get(), false);
```

---

### Bug 3: CheckBmmOp 中 CheckDtypeValid 被重复调用

- **位置**: 第 750-761 行, `CheckBmmOp` 函数
- **类型**: 逻辑错误 (冗余校验/潜在冲突)
- **严重程度**: 中等 (Medium)
- **描述**: 当 `mat2->GetStorageFormat() == FORMAT_FRACTAL_NZ` 时，先调用了 `CheckDtypeValidWeightNz`（仅支持 FP16/BF16），然后又无条件调用了 `CheckDtypeValid`（支持 FP32/FP16/BF16）。两个函数对dtype的校验规则不同，可能导致：1) 冗余校验降低性能；2) 如果意图是互斥校验，则 NZ 格式下 FP32 输入本应被 `CheckDtypeValidWeightNz` 拒绝，但控制流仍然通过第二次 `CheckDtypeValid` 不会报错（因为已经在第一个 CHECK_RET 返回了）。更严重的是，非 NZ 格式时，`CheckDtypeValid` 被调用了两次（line 757 和 line 759）。
- **触发条件**: 任何非 NZ 格式的输入都会导致 `CheckDtypeValid` 被执行两次。
- **测试方案**: 传入 ND 格式的 tensor，观察 `CheckDtypeValid` 是否被调用两次（通过日志或断点）。

```cpp
// 缺陷代码 (第754-759行):
if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
    CHECK_RET(CheckDtypeValidWeightNz(self, mat2, out), ACLNN_ERR_PARAM_INVALID);
} else {
    CHECK_RET(CheckDtypeValid(self, mat2, bias, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);
}
CHECK_RET(CheckDtypeValid(self, mat2, bias, out, cubeMathType), ACLNN_ERR_PARAM_INVALID); // 多余

// 修复: 删除第759行的重复调用
```

---

### Bug 4: CheckBmmResIsEmpty 缺少 K 维度为0的检查

- **位置**: 第 104-108 行, `CheckBmmResIsEmpty` 函数
- **类型**: 边界条件遗漏
- **严重程度**: 中等 (Medium)
- **描述**: 对于 [B,M,K] @ [B,K,N] = [B,M,N] 的 BMM，当 K=0 时（`self->GetViewShape().GetDim(THIRD_DIM) == 0`），运算结果应为全零矩阵。但当前函数仅检查了 B=0、M=0、N=0 的情况，遗漏了 K=0。虽然 K=0 时 self 本身为 empty tensor，可能在其他地方被 `IsEmpty()` 捕获，但在 `CreateBatchMatmulGraphImpl` 中（结合 Bug 1），这种情况不会被正确处理。
- **触发条件**: 输入 self 的 shape 为 [B, M, 0]，mat2 的 shape 为 [B, 0, N]，且 B、M、N 均非零。
- **测试方案**: 构造 self=[2,3,0], mat2=[2,0,4] 的输入，验证输出是否为全零的 [2,3,4] tensor。

```cpp
// 缺陷代码:
static inline bool CheckBmmResIsEmpty(const aclTensor* self, const aclTensor* mat2)
{
    return self->GetViewShape().GetDim(FIRST_DIM) == 0 || self->GetViewShape().GetDim(SECOND_DIM) == 0 ||
        mat2->GetViewShape().GetDim(THIRD_DIM) == 0;
}

// 修复: 增加 K 维度检查
static inline bool CheckBmmResIsEmpty(const aclTensor* self, const aclTensor* mat2)
{
    return self->GetViewShape().GetDim(FIRST_DIM) == 0 || self->GetViewShape().GetDim(SECOND_DIM) == 0 ||
        self->GetViewShape().GetDim(THIRD_DIM) == 0 || mat2->GetViewShape().GetDim(THIRD_DIM) == 0;
}
```

---

### Bug 5: contiguousMat2 使用 SetStorageShape 修改 const 指针对象

- **位置**: 第 897 行
- **类型**: 类型安全违规 (const 正确性)
- **严重程度**: 低 (Low)
- **描述**: `contiguousMat2` 在第 862 行被声明为 `const aclTensor*`（在某些分支赋值后），但在第 897 行通过 `contiguousMat2->SetStorageShape(...)` 直接调用了非 const 成员方法。这依赖于编译器对 `SetStorageShape` 是否为 const 方法的判定，若非 const 则存在未定义行为或需要 `const_cast`。结合上下文（第 880 行有类似的 `const_cast` 用法），此处缺少显式 `const_cast`。
- **触发条件**: mat2 为 FORMAT_FRACTAL_NZ 格式时触发。
- **测试方案**: 使用 NZ 格式 mat2 输入，检查 storageShape 是否被正确刷新，以及是否有编译 warning。

---

### Bug 6: ProcessEmptyTensor 使用 self 的 dtype 而非推导后的输出 dtype

- **位置**: 第 252-268 行, `ProcessEmptyTensor` 函数
- **类型**: 类型推导错误
- **严重程度**: 低 (Low)
- **描述**: `ProcessEmptyTensor` 中 `executor->AllocTensor(bmmEmptyShape, self->GetDataType())` 使用 self 的数据类型作为输出 tensor 的类型。但 BMM 的输出 dtype 可能与 self 不同（例如 self 为 FP16，out 为 FP32）。应使用 out 的 dtype 或经过类型推导后的 dtype。该函数仅在旧路径 `ExecBmmOpWithBias` 中被调用，且后续有 Cast 操作，但如果输出为 empty（第257行提前返回），则跳过了 Cast，类型可能不匹配。
- **触发条件**: self dtype 为 FP16，out dtype 为 FP32，且输出 shape 的某维为 0（全空）。
- **测试方案**: 构造 FP16 self + FP32 out + M=0 的场景，检查输出 tensor 的 dtype 是否正确。

---

## 汇总表

| 编号 | 位置 (行号) | Bug 类型 | 严重程度 | 简要描述 |
|------|------------|----------|----------|----------|
| 1 | 985-989 | 逻辑错误 | 严重 | 空tensor分支被无条件覆盖，缺少else |
| 2 | 1045 | 参数缺失 | 严重 | ExecBmmOp调用缺少isBaddbmm参数 |
| 3 | 754-759 | 冗余校验 | 中等 | CheckDtypeValid被重复调用 |
| 4 | 104-108 | 边界条件 | 中等 | CheckBmmResIsEmpty遗漏K=0检查 |
| 5 | 897 | const正确性 | 低 | 对const指针对象调用非const方法 |
| 6 | 252-268 | 类型推导 | 低 | 空tensor输出使用self的dtype而非out的dtype |
