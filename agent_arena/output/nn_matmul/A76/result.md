# aclnn_matmul.cpp 代码审查报告

## Bug 列表

### Bug 1: MatMulDimNumMatBGe3Graph 中 Unsqueeze 维度条件反转

- **位置**: 第 734-738 行, `MatMulDimNumMatBGe3Graph::PreProcess()`
- **类型**: 逻辑错误
- **严重程度**: 严重 (Critical)
- **描述**: 当 `dimTensor1 == 1` 时，代码使用 `dimData = {0}` 进行 unsqueeze，将 `[m]` 变为 `[1, m]`（2D）；当 `dimTensor1 == 2` 时，使用 `dimData = {0, 1}`，将 `[n, m]` 变为 `[1, 1, n, m]`（4D）。然而 matB 为 3D（`[B, m, p]`），BatchMatmul 需要两个输入维度匹配。正确逻辑应为：
  - `dimTensor1 == 1`: 需要增加2个维度 → `dimData = {0, 1}`，使 `[m]` → `[1, 1, m]`（3D）
  - `dimTensor1 == 2`: 需要增加1个维度 → `dimData = {0}`，使 `[n, m]` → `[1, n, m]`（3D）

  条件分支的 dimData 赋值逻辑完全反转。
- **触发条件**: 当 self 为 1D 或 2D 且 mat2 为 3D 时触发。例如 `self=[m]`, `mat2=[B, m, p]` 或 `self=[n, m]`, `mat2=[B, m, p]`。
- **测试方案**:
  ```python
  # Case 1: 1D x 3D
  self = torch.randn(4)           # [4]
  mat2 = torch.randn(2, 4, 5)    # [2, 4, 5]
  out = torch.matmul(self, mat2)  # 期望 [2, 5], 实际维度不匹配报错

  # Case 2: 2D x 3D
  self = torch.randn(3, 4)       # [3, 4]
  mat2 = torch.randn(2, 4, 5)   # [2, 4, 5]
  out = torch.matmul(self, mat2) # 期望 [2, 3, 5], 实际维度不匹配报错
  ```

---

### Bug 2: ProcessEmptyTensor 使用错误的数据类型分配输出张量

- **位置**: 第 255 行, `ProcessEmptyTensor()` 函数
- **类型**: 数据类型错误
- **严重程度**: 中等 (Medium)
- **描述**: `executor->AllocTensor(outShape, self->GetDataType())` 使用了输入 `self` 的数据类型来分配输出张量，但输出张量的数据类型应该与 `out` 参数一致。当 self 和 out 的数据类型不同时（如 self 为 fp16，out 为 fp32），输出张量的类型将不正确。同样的问题也存在于 `MatMulEmptyTensorGraph::Impl()`（第 522 行）。
- **触发条件**: 当输入为空张量且 self 的 dtype 与 out 的 dtype 不同时触发。例如 self 为 float16 空张量，out 指定为 float32。
- **测试方案**:
  ```python
  # 创建空张量，self和out dtype不同
  self = torch.empty(0, 4, dtype=torch.float16)
  mat2 = torch.randn(4, 5, dtype=torch.float16)
  out = torch.empty(0, 5, dtype=torch.float32)
  # 检查输出tensor的dtype是否为float32
  aclnn_matmul(self, mat2, out)
  assert out.dtype == torch.float32  # 实际可能得到float16
  ```

---

### Bug 3: const_cast 修改输入张量状态导致潜在副作用

- **位置**: 第 466 行 (`BuildMatMulWeightNzGraph`) 和第 794 行 (`MatMulWeightNzGraph::Impl`)
- **类型**: 数据完整性/副作用问题
- **严重程度**: 中等 (Medium)
- **描述**: `const_cast<aclTensor *>(mat2)->SetViewShape(SwapLastTwoDimValue(mat2->GetViewShape()))` 通过 const_cast 强制移除 const 限定符并修改了输入张量的 ViewShape。这违反了函数签名中 `const aclTensor* mat2` 的 const 契约，修改了调用者传入的张量状态。如果调用方在此函数返回后继续使用 mat2，将观察到被修改的 shape，可能导致后续计算错误。
- **触发条件**: 当 mat2 为 NZ 格式且检测到转置时（即 stride 表示转置布局），mat2 的 ViewShape 被永久修改。
- **测试方案**:
  ```cpp
  // 创建 NZ 格式的 mat2 并标记为转置
  auto mat2 = CreateNZTensor({K, N}, transposed_strides);
  auto originalShape = mat2->GetViewShape();
  BuildMatMulWeightNzGraph(self, mat2, out, cubeMathType, executor);
  // 验证 mat2 的 shape 是否被意外修改
  ASSERT_EQ(mat2->GetViewShape(), originalShape); // 将失败
  ```

---

### Bug 4: 常量命名误导 NZ_K0_VALUE_32 实际值为 8

- **位置**: 第 53 行
- **类型**: 命名规范/可维护性
- **严重程度**: 低 (Low)
- **描述**: `static const int NZ_K0_VALUE_32 = 8;` 常量名中的 "32" 暗示值为 32，但实际值为 8。虽然在 NZ 格式中 float32 数据类型的 k0 值确实是 8（因为 16 个 float16 等价于 8 个 float32 的存储空间），但命名中使用 "32" 容易让维护者误解。对比 `NZ_K0_VALUE_16 = 16` 的命名模式更加混乱。建议重命名为 `NZ_K0_VALUE_FP32 = 8` 或 `NZ_K0_DIM_FOR_FLOAT32 = 8`。
- **触发条件**: 不影响运行时行为，但在代码维护和修改时可能导致开发者误用。
- **测试方案**: 代码审查和静态分析即可发现。验证所有使用该常量的场景中 float32 的 k0 值确实为 8。

---

### Bug 5: aclnnMatmulWeightNzGetWorkspaceSize 中参数校验和资源创建顺序不当

- **位置**: 第 929-935 行
- **类型**: 资源管理/性能
- **严重程度**: 低 (Low)
- **描述**: 在 `aclnnMatmulWeightNzGetWorkspaceSize` 中，先调用 `CREATE_EXECUTOR()`（第 930 行）再进行参数校验 `CheckWeightNzParam`（第 934 行）。对比 `aclnnMatmulGetWorkspaceSize`（第 887-892 行）中先校验参数再创建 executor 的正确模式。当参数非法时，executor 被不必要地创建后再因函数返回而销毁，浪费资源。虽然智能指针确保不会内存泄漏，但违反了 fail-fast 原则且增加了不必要的开销。
- **触发条件**: 当传入非法参数时（如空指针、不支持的 dtype 等），在 WeightNz 路径中会先创建 executor 再失败返回。
- **测试方案**:
  ```cpp
  // 传入非法参数，对比两个API的行为
  auto ret = aclnnMatmulWeightNzGetWorkspaceSize(nullptr, mat2, out, 0, &ws, &exec);
  // 验证返回错误码，检查是否有不必要的资源分配(通过性能profiling)
  ```

---

### Bug 6: CheckShapeValid 对 dimTensor1==1 && dimTensor2==1 场景的 K 轴校验缺失独立分支

- **位置**: 第 180 行
- **类型**: 逻辑不清晰/潜在遗漏
- **严重程度**: 低 (Low)
- **描述**: 当 `dimTensor1 == 1 && dimTensor2 == 1`（向量点积场景）时，代码落入 `dimTensor2 == 1 || dimTensor2 == 2` 分支。此分支的注释写着 "tensor1 dims number is 1 OR tensor2 dims number is 2"，与实际条件 `dimTensor2 == 1 || dimTensor2 == 2` 不符（注释提到 tensor1 但代码检查的是 tensor2）。虽然对于 dot product 场景（1D x 1D），`selfShape.GetDim(dimTensor1 - 1) == mat2Shape.GetDim(0)` 即 `dim(0) == dim(0)` 的校验逻辑是正确的，但注释的错误说明可能掩盖了原始设计意图中的条件遗漏。
- **触发条件**: 不直接影响运行时，但可能在后续代码修改时引入问题。
- **测试方案**: 验证 1D x 1D 维度不匹配时能正确报错：`self=[3]`, `mat2=[4]` 应返回错误。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| Bug 1 | 第 734-738 行 | 逻辑错误 | 严重 | Unsqueeze 维度条件反转，dimTensor1==1 和 dimTensor1==2 的分支赋值互换 |
| Bug 2 | 第 255, 522 行 | 数据类型错误 | 中等 | 空张量输出使用 self 的 dtype 而非 out 的 dtype |
| Bug 3 | 第 466, 794 行 | 副作用问题 | 中等 | const_cast 修改 const 输入张量的 ViewShape，破坏调用者数据 |
| Bug 4 | 第 53 行 | 命名规范 | 低 | NZ_K0_VALUE_32=8 命名误导 |
| Bug 5 | 第 929-935 行 | 资源管理 | 低 | 参数校验应在 executor 创建之前执行 |
| Bug 6 | 第 180 行 | 注释/逻辑不清晰 | 低 | 条件分支注释与代码不匹配 |
