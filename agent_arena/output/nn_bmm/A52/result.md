# Ascend NPU 算子代码审查报告

**文件**: `aclnn_batch_matmul.cpp`  
**审查范围**: 参数校验、类型推导、边界条件、错误处理、逻辑正确性

---

### Bug 1: `CreateBatchMatmulGraphImpl` 空tensor分支被覆盖（死代码）

- **位置**: 第 974-985 行
- **类型**: 逻辑错误
- **严重程度**: 严重 (Critical)
- **描述**: 函数在 `CheckBmmResIsEmpty` 为 true 时创建了 `BatchmmEmptyTensorGraph`，但紧接着无条件地将 `matmulGraph` 覆盖为 `BatchMatmulExecBmmOpGraph`。空tensor分支永远不会生效，导致空tensor场景下执行了非法的计算图。
- **触发条件**: 输入 self 或 mat2 的 batch/M/N 维度为 0 时触发（空tensor场景）。
- **测试方案**: 构造 shape 为 `[0, M, K]` 或 `[B, 0, K]` 或 `[B, K, 0]` 的输入tensor调用 `aclnnBatchMatMulGetWorkspaceSize`，验证是否正确处理空tensor而非崩溃。

**修复建议**:
```cpp
if (CheckBmmResIsEmpty(self, mat2)) {
    matmulGraph = std::make_shared<BatchmmEmptyTensorGraph>(...);
    return matmulGraph;  // 缺少 return
}
matmulGraph = std::make_shared<BatchMatmulExecBmmOpGraph>(...);
return matmulGraph;
```

---

### Bug 2: `ExecBmmOp` 调用缺少参数 `isBaddbmm`

- **位置**: 第 1046 行
- **类型**: 编译错误 / 参数缺失
- **严重程度**: 严重 (Critical)
- **描述**: `ExecBmmOp` 函数签名（第914行）需要 6 个参数 `(self, mat2, out, cubeMathType, executor, isBaddbmm)`，但在 `aclnnBatchMatMulWeightNzGetWorkspaceSize` 中调用时只传了 5 个参数，缺少 `isBaddbmm`。若编译器未提供默认值则会导致编译失败；若头文件中声明了默认值则可能传入错误值。
- **触发条件**: 编译或调用 `aclnnBatchMatMulWeightNzGetWorkspaceSize` 接口。
- **测试方案**: 直接编译此文件，观察是否报错；若能编译，则验证 WeightNz 路径下 `isBaddbmm` 行为是否符合预期。

**修复建议**:
```cpp
auto bmmOut = ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get(), false);
```

---

### Bug 3: `CheckBmmResIsEmpty` 未校验维度数，直接硬编码索引访问

- **位置**: 第 104-108 行
- **类型**: 边界条件错误
- **严重程度**: 中等 (Medium)
- **描述**: 函数直接使用 `FIRST_DIM(0)`、`SECOND_DIM(1)`、`THIRD_DIM(2)` 索引访问 shape，但未验证 tensor 维度数是否 >= 3。如果在 `CheckShape` 之前被调用且 tensor 维度 < 3，则会越界访问导致未定义行为。在 `aclnnBatchMatMulGetWorkspaceSize`（第1003行）中，`CheckBmmResIsEmpty` 在 `CheckParamsV2`（含维度校验）之后被调用，顺序正确；但 `ExecBmmOpWithBias`（第818行）中在 `CheckBmmOp` 内无明确 shape 维度校验的情况下就调用了。
- **触发条件**: 输入 tensor 维度小于 3 时。
- **测试方案**: 传入 2D tensor `[M, K]` 调用 `ExecBmmOpWithBias`，观察是否越界崩溃。

---

### Bug 4: `ProcessEmptyTensor` 硬编码3D索引，不兼容高维tensor

- **位置**: 第 248-249 行
- **类型**: 边界条件错误
- **严重程度**: 中等 (Medium)
- **描述**: `ProcessEmptyTensor` 使用 `(self->GetViewShape())[0]`、`[1]`、`(mat2->GetViewShape())[2]` 构造输出 shape，假定输入为严格 3D。如果输入为 4D/5D/6D（BMM支持的维度），则输出 shape 计算错误，丢失高维 batch 信息。
- **触发条件**: 输入为4D及以上高维空tensor（如 `[2, 0, M, K]`）时输出shape错误。
- **测试方案**: 构造 `[2, 3, 0, K]` shape 的 self tensor 和对应 mat2，验证空tensor输出shape是否正确包含所有batch维。

---

### Bug 5: `contiguousMat2` 为 `const aclTensor*` 却调用非const方法

- **位置**: 第 891 行
- **类型**: const正确性违规
- **严重程度**: 中等 (Medium)
- **描述**: `contiguousMat2` 的类型推导自 `const aclTensor*`（第856行 `auto contiguousMat2 = reformatMat2;`，而 `reformatMat2` 是 `const aclTensor*`）。第891行直接调用 `contiguousMat2->SetStorageShape(...)` 是在const指针上调用修改方法，除非编译器做了隐式转换或 `SetStorageShape` 被标记为 const（语义上不合理），否则会编译失败或导致未定义行为。
- **触发条件**: `mat2` 为 `FORMAT_FRACTAL_NZ` 格式时执行到该行。
- **测试方案**: 以 NZ 格式的 mat2 输入调用 BMM 接口，验证 StorageShape 是否正确设置。

---

### Bug 6: `CheckBmmOp` 中 `CheckDtypeValid` 被重复调用

- **位置**: 第 748-753 行
- **类型**: 逻辑冗余 / 潜在错误
- **严重程度**: 低 (Low)
- **描述**: 当 `mat2` 为 `FORMAT_FRACTAL_NZ` 时，先调用 `CheckDtypeValidWeightNz`，然后在第753行又无条件调用 `CheckDtypeValid`。对 WeightNz 场景，`CheckDtypeValid` 的检查逻辑（支持 DT_FLOAT）与 `CheckDtypeValidWeightNz`（仅支持 FP16/BF16）矛盾，会导致 NZ 路径上 DT_FLOAT 被错误放行或因逻辑不一致产生误判。
- **触发条件**: mat2 为 NZ 格式、输入为 DT_FLOAT16 时，两个检查都通过但逻辑冗余；若传入非预期类型可能出现不一致。
- **测试方案**: 以 NZ 格式且 dtype 为 DT_FLOAT 的 mat2 调用，检查两个校验函数的返回值是否一致。

---

### Bug 7: `CheckTransNonContiguousShapeSupport` 中 `adjX2` 未考虑导致 nDim 计算可能错误

- **位置**: 第 777 行
- **类型**: 逻辑错误
- **严重程度**: 低 (Low)
- **描述**: 注释写明"非连续场景viewshape一定是bkn格式"，所以直接取最后一维作为 nDim。但函数签名中没有 `adjX2` 参数，如果存在 mat2 转置的非连续场景（即 `isNeedSwapInnerTwoDim == true`），mat2 的 viewshape 可能仍为 `[B, N, K]` 格式，此时取最后一维得到的是 K 而非 N。
- **触发条件**: mat2 为非连续且内轴需要交换的场景。
- **测试方案**: 构造 mat2 为转置非连续状态（stride表示转置），验证 shape 判断逻辑是否正确获取 N 维度。

---

## 汇总表

| # | 位置 | Bug类型 | 严重程度 | 简要描述 |
|---|------|---------|----------|----------|
| 1 | L979-984 | 逻辑错误 | Critical | `CreateBatchMatmulGraphImpl` 空tensor分支被无条件覆盖，缺少 return/else |
| 2 | L1046 | 参数缺失 | Critical | `ExecBmmOp` 调用缺少第6个参数 `isBaddbmm` |
| 3 | L104-108 | 边界条件 | Medium | `CheckBmmResIsEmpty` 未校验维度数直接按3D索引访问 |
| 4 | L248-249 | 边界条件 | Medium | `ProcessEmptyTensor` 硬编码3D，不支持高维输入 |
| 5 | L891 | const违规 | Medium | const指针上调用非const修改方法 |
| 6 | L748-753 | 逻辑冗余 | Low | `CheckDtypeValid` 在NZ分支被重复/矛盾调用 |
| 7 | L777 | 逻辑错误 | Low | 非连续转置场景nDim取值可能错误 |
