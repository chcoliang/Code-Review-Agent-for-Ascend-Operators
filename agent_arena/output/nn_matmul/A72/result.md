# aclnn_matmul.cpp 代码审查报告

## Bug 列表

### Bug 1: ProcessEmptyTensor 使用错误的数据类型分配输出张量

- **位置**: 第 255 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `ProcessEmptyTensor` 函数中使用 `self->GetDataType()` 分配输出张量的数据类型，但正确的做法应该是使用 `out->GetDataType()`。当 self 和 out 的数据类型不一致时（例如 self 为 fp16，out 为 fp32），输出张量的 dtype 将与期望不符，导致后续计算结果错误或内存访问异常。
- **触发条件**: self 的数据类型与 out 的数据类型不同时（例如 self=DT_FLOAT16, out=DT_FLOAT），且 self 或 mat2 为空张量。
- **测试方案**: 构造 self 为 fp16 空张量（shape 含 0 维），out 为 fp32 张量，调用 matmul，验证输出 dtype 是否正确为 fp32。

---

### Bug 2: aclnnMatmulWeightNzGetWorkspaceSize 中 Executor 在参数校验前创建

- **位置**: 第 922-927 行
- **类型**: 资源管理/执行顺序错误
- **严重程度**: 中
- **描述**: 在 `aclnnMatmulWeightNzGetWorkspaceSize` 中，`CREATE_EXECUTOR()` 在参数校验 `CheckWeightNzParam` 之前调用（第 923 行创建，第 927 行校验）。而对比 `aclnnMatmulGetWorkspaceSize`（第 887 行先校验，第 891 行再创建），顺序是正确的。当参数无效时，executor 已经被创建但未被正确释放，造成资源浪费；同时违反了代码中注释的"固定写法"模式。
- **触发条件**: 调用 `aclnnMatmulWeightNzGetWorkspaceSize` 并传入无效参数（如空指针、不支持的 dtype）。
- **测试方案**: 传入 nullptr 或不支持的 dtype 调用该接口，检查是否有内存泄漏（通过内存分析工具如 ASAN 检测）。

---

### Bug 3: MatMulDimNumMatBGe3Graph 中 dimTensor1==2 时过度 Unsqueeze

- **位置**: 第 737-738 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 当 `dimTensor1 == 2` 时，代码对 matA 执行 `UnsqueezeNd` 的维度为 `{0, 1}`，将 2D [M, K] 变为 4D [1, 1, M, K]。但此分支条件为 `dimTensor1 == 1 || dimTensor1 == 2` 且 `dimTensor2 >= 3`。当 matB 为 3D [B, K, N] 时，matA 变为 [1, 1, M, K]（2个batch维） vs matB [B, K, N]（1个batch维），batch 维度数不匹配可能导致 BatchMatmul 内部广播或计算错误。正确做法是对 dimTensor1==2 的情况仅 unsqueeze 在 `{0}` 得到 [1, M, K]。
- **触发条件**: self 为 2D 张量（如 [M, K]），mat2 为 3D 张量（如 [B, K, N]）进行矩阵乘法。
- **测试方案**: 构造 self shape=[4, 8], mat2 shape=[3, 8, 16]，执行 matmul，验证结果 shape 为 [3, 4, 16] 且数值正确。

---

### Bug 4: const_cast 修改输入张量的 ViewShape 导致副作用

- **位置**: 第 466 行, 第 794 行
- **类型**: 数据完整性/副作用错误
- **严重程度**: 中
- **描述**: `const_cast<aclTensor*>(mat2)->SetViewShape(SwapLastTwoDimValue(mat2->GetViewShape()))` 直接修改了传入的 const 指针所指向的张量的 ViewShape。这会影响调用者持有的 mat2 张量状态，违反了 const 语义契约。如果调用方在此函数返回后继续使用 mat2，将看到被篡改的 shape。在 `MatMulWeightNzGraph::Impl()`（第 794 行）中同样存在此问题。
- **触发条件**: mat2 为 NZ 格式的转置张量时，调用 `BuildMatMulWeightNzGraph` 或 `MatMulWeightNzGraph::Impl()` 后，外部代码继续使用 mat2 张量。
- **测试方案**: 在调用 matmul WeightNz 接口后，检查 mat2 的 ViewShape 是否被意外修改（对比调用前后的 shape）。

---

### Bug 5: CreateMatmulGraphImpl 中注释与代码条件不匹配

- **位置**: 第 845 行
- **类型**: 注释错误/可维护性
- **严重程度**: 低
- **描述**: 第 845 行的注释为 `// dimTensor1 is 1 or 2 && dimTensor2 >= 3`，但实际代码条件为 `dimTensor1 >= 3 && (dimTensor2 == 1 || dimTensor2 == 2)`，注释描述的是完全相反的情况。此注释与第 848 行的注释内容互换了，容易误导后续维护者。
- **触发条件**: 代码维护时依赖注释理解逻辑分支。
- **测试方案**: 代码评审检查，确认注释与条件一致。

---

### Bug 6: CheckShapeValid 中 dimTensor1==1 时 K 轴校验条件分支不完整

- **位置**: 第 180 行
- **类型**: 逻辑缺陷
- **严重程度**: 低
- **描述**: `else if (dimTensor2 == 1 || dimTensor2 == 2)` 分支的注释写的是 "tensor1 dims number is 1 OR tensor2 dims number is 2"，但实际上此分支无论 dimTensor1 值如何都会进入（只要 dimTensor2==1 或 2）。当 dimTensor1 >= 3 且 dimTensor2 == 2 时，selfKDim 取 `selfShape.GetDim(dimTensor1 - 1)` 即最后一维，这对于高维 self 是正确的（K轴在最后一维），但注释误导了代码意图。
- **触发条件**: N/A（功能上正确，但注释错误）。
- **测试方案**: 代码评审。

---

### Bug 7: NZ_K0_VALUE_32 常量命名与值不一致

- **位置**: 第 53 行
- **类型**: 命名规范/可维护性
- **严重程度**: 低
- **描述**: `NZ_K0_VALUE_32 = 8`，常量名中含 "32" 但值为 8。虽然 "32" 指代 32-bit 数据类型（FP32 在 NZ 格式下 c0=8），但命名极易与值 32 混淆。结合第 52 行 `NZ_K0_VALUE_16 = 16`（16-bit 类型，c0=16），命名模式不统一，后续开发者可能误用。
- **触发条件**: 开发者误将 NZ_K0_VALUE_32 当作值为 32 的常量使用。
- **测试方案**: 代码评审，建议重命名为 `NZ_K0_VALUE_FP32 = 8` 或 `NZ_C0_FOR_32BIT = 8`。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第 255 行 | 逻辑错误 | 高 | 空张量处理使用 self dtype 而非 out dtype 分配输出 |
| 2 | 第 922-927 行 | 资源管理 | 中 | Executor 在参数校验前创建，校验失败时资源浪费 |
| 3 | 第 737-738 行 | 逻辑错误 | 高 | dimTensor1==2 时多余 unsqueeze 导致 batch 维度不匹配 |
| 4 | 第 466/794 行 | 副作用错误 | 中 | const_cast 修改输入张量 shape，破坏调用方数据 |
| 5 | 第 845 行 | 注释错误 | 低 | 注释与代码条件完全相反 |
| 6 | 第 180 行 | 注释错误 | 低 | 分支注释描述不准确 |
| 7 | 第 53 行 | 命名规范 | 低 | 常量名暗示值为 32 实际为 8 |
