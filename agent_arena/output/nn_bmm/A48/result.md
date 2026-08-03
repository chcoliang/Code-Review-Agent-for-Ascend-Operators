# Ascend NPU BatchMatMul 算子代码审查报告

## Bug 列表

### Bug 1: `CreateBatchMatmulGraphImpl` 空tensor分支缺少 `return`，逻辑被覆盖

- **位置**: 第 980-991 行，`CreateBatchMatmulGraphImpl` 函数
- **类型**: 逻辑错误 / 控制流缺陷
- **严重程度**: 严重 (Critical)
- **描述**: 当 `CheckBmmResIsEmpty` 为 true 时，创建了 `BatchmmEmptyTensorGraph` 并赋值给 `matmulGraph`，但紧接着无条件地又创建了 `BatchMatmulExecBmmOpGraph` 并覆盖了该指针。空tensor分支永远不会生效。
- **触发条件**: 输入 self 的 batch=0 或 M=0，或 mat2 的 N=0 时，本应返回空tensor图，但实际会执行正常的BMM计算图，导致对空tensor执行无效计算或崩溃。
- **测试方案**: 构造 shape 为 [0, M, K] 或 [B, 0, K] 的输入 tensor，调用 `aclnnBatchMatMulGetWorkspaceSize`，验证是否正确跳过计算。

```cpp
// 错误代码 (line 983-991):
if (CheckBmmResIsEmpty(self, mat2)) {
    matmulGraph = std::make_shared<BatchmmEmptyTensorGraph>(...);
}
// 缺少 else，下面的赋值无条件覆盖
matmulGraph = std::make_shared<BatchMatmulExecBmmOpGraph>(...);

// 修复：添加 else 或在 if 分支内 return
if (CheckBmmResIsEmpty(self, mat2)) {
    matmulGraph = std::make_shared<BatchmmEmptyTensorGraph>(...);
} else {
    matmulGraph = std::make_shared<BatchMatmulExecBmmOpGraph>(...);
}
```

---

### Bug 2: `ExecBmmOp` 调用缺少必需参数 `isBaddbmm`

- **位置**: 第 1052 行，`aclnnBatchMatMulWeightNzGetWorkspaceSize` 函数内
- **类型**: 参数缺失 / 编译错误
- **严重程度**: 严重 (Critical)
- **描述**: `ExecBmmOp` 函数签名（第 920-921 行）需要 6 个参数：`(self, mat2, out, cubeMathType, executor, isBaddbmm)`，但第 1052 行调用时只传了 5 个参数，缺少 `isBaddbmm`。
- **触发条件**: 编译该文件时会触发编译错误（除非有重载或默认参数在头文件中声明）。如果头文件中有默认参数，则不是编译错误但属于与本文件定义不一致的问题。
- **测试方案**: 直接编译该文件，检查是否有参数数量不匹配的编译报错。

```cpp
// 错误代码 (line 1052):
auto bmmOut = ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get());

// 修复：补充 isBaddbmm 参数
auto bmmOut = ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get(), false);
```

---

### Bug 3: `CheckBmmOp` 中对非 NZ 格式重复调用 `CheckDtypeValid`

- **位置**: 第 750-761 行，`CheckBmmOp` 函数
- **类型**: 逻辑冗余 / 潜在错误
- **严重程度**: 中等 (Medium)
- **描述**: 当 `mat2` 格式为 `FORMAT_FRACTAL_NZ` 时，先调用 `CheckDtypeValidWeightNz`，然后在第 759 行又无条件调用了 `CheckDtypeValid`。对于 NZ 格式，`CheckDtypeValidWeightNz` 要求 self/mat2/out 数据类型完全一致且仅支持 FP16/BF16，但随后的 `CheckDtypeValid` 可能对合法的 BF16 NZ 输入产生错误的警告日志（dtypeMatch 检查等）。更重要的是，非 NZ 路径会调用两次 `CheckDtypeValid`，造成冗余。
- **触发条件**: mat2 为 FRACTAL_NZ 格式且数据类型为 BF16 时，`CheckDtypeValid` 可能打印不必要的警告或在特定 SOC 版本下误报。
- **测试方案**: 使用 NZ 格式的 BF16 mat2 输入，验证 check 函数是否产生误报或矛盾的校验逻辑。

```cpp
// 错误代码 (line 754-759):
if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
    CHECK_RET(CheckDtypeValidWeightNz(self, mat2, out), ACLNN_ERR_PARAM_INVALID);
} else {
    CHECK_RET(CheckDtypeValid(self, mat2, bias, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);
}
CHECK_RET(CheckDtypeValid(self, mat2, bias, out, cubeMathType), ACLNN_ERR_PARAM_INVALID); // 多余

// 修复：删除第 759 行的重复调用
```

---

### Bug 4: `const aclTensor*` 指针调用非 const 方法 `SetStorageShape`

- **位置**: 第 897 行，`ExecBmmOpWithBias` 函数
- **类型**: const 正确性违反 / 潜在编译错误
- **严重程度**: 中等 (Medium)
- **描述**: `contiguousMat2` 类型为 `const aclTensor*`（由第 862 行的 `auto` 推导自 `reformatMat2`），在第 897 行直接调用 `contiguousMat2->SetStorageShape()`，这是一个非 const 方法，会违反 const 约束。同一函数中第 880 行对类似操作使用了 `const_cast`，但此处遗漏了。
- **触发条件**: 编译时可能产生错误或警告（取决于 `SetStorageShape` 的声明是否为 const——如果是 mutable 内部状态则不会报错，但语义上仍然不安全）。
- **测试方案**: 严格编译该文件，开启 `-Werror` 检查是否有 const 违规警告。

```cpp
// 错误代码 (line 897):
contiguousMat2->SetStorageShape(mat2->GetStorageShape());

// 修复：
const_cast<aclTensor*>(contiguousMat2)->SetStorageShape(mat2->GetStorageShape());
```

---

### Bug 5: `CheckBmmResIsEmpty` 硬编码维度索引假设输入为 3D

- **位置**: 第 104-108 行，`CheckBmmResIsEmpty` 函数
- **类型**: 健壮性缺陷
- **严重程度**: 低 (Low)
- **描述**: 函数使用 `FIRST_DIM(0)`, `SECOND_DIM(1)`, `THIRD_DIM(2)` 硬编码索引检查 shape，但注释说明适用于 `[B,M,K]@[B,K,N]`，隐含假设输入严格为 3D。虽然当前调用点（`aclnnBatchMatMulGetWorkspaceSize`）在调用前通过 `CheckParamsV2` -> `CheckShape` 强制了 3D，但 `CreateBatchMatmulGraphImpl` 也调用它，若未来函数被复用到非 3D 场景会产生错误。
- **触发条件**: 若其他代码路径在未经 3D 校验时调用此函数（如 4D+ tensor），会检查错误的维度。
- **测试方案**: 构造 4D 输入 [B1, B2, M, K]，绕过 shape 检查直接调用 `CheckBmmResIsEmpty`，验证返回值是否正确。

---

### Bug 6: `LAST_DIM` 和 `PENULTIMATE_DIM` 常量命名与值易混淆

- **位置**: 第 71-72 行，常量定义
- **类型**: 可维护性 / 潜在误用
- **严重程度**: 低 (Low)
- **描述**: `PENULTIMATE_DIM = 2` 和 `LAST_DIM = 1` 作为从尾部的偏移量使用（如 `dimNum - PENULTIMATE_DIM` 表示倒数第二维），但命名容易与"第一维/第二维的索引"混淆。在 `SwapLastTwoDimValue(mat2->GetViewShape(), LAST_DIM, PENULTIMATE_DIM)` 调用中（第 876 行），如果该函数期望的是绝对索引而非偏移量，则会产生错误结果。
- **触发条件**: 当 `SwapLastTwoDimValue` 接口变更或被误解时，传入 `LAST_DIM=1, PENULTIMATE_DIM=2` 可能导致交换错误的维度。
- **测试方案**: 检查 `SwapLastTwoDimValue` 的函数签名，确认参数语义是偏移量还是绝对索引。

---

## 汇总表

| 编号 | 位置 (行) | Bug 类型 | 严重程度 | 简要描述 |
|------|-----------|----------|----------|----------|
| 1 | 980-991 | 逻辑错误 | 严重 | 空tensor分支缺少return/else，被无条件覆盖 |
| 2 | 1052 | 参数缺失 | 严重 | `ExecBmmOp`调用缺少`isBaddbmm`参数 |
| 3 | 754-759 | 逻辑冗余 | 中等 | NZ格式下重复调用`CheckDtypeValid` |
| 4 | 897 | const违反 | 中等 | const指针直接调用非const方法 |
| 5 | 104-108 | 健壮性 | 低 | 硬编码3D维度索引，缺乏通用性 |
| 6 | 71-72 | 可维护性 | 低 | 常量命名易混淆偏移与索引语义 |
