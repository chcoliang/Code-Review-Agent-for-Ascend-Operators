# Ascend NPU 算子代码审查报告 - aclnn_matmul.cpp

## Bug 列表

### Bug 1: MatMulDimNumMatBGe3Graph 中 Unsqueeze 维度逻辑交换错误

- **位置**: 第 733-737 行，`MatMulDimNumMatBGe3Graph::PreProcess()`
- **类型**: 逻辑错误
- **严重程度**: 严重 (Critical)
- **描述**: 当 `dimTensor1 == 1` 时（self 为 1D 张量 [K]），代码使用 `dimData = {0}` 仅添加一个维度变为 [1, K]（2D）；当 `dimTensor1 == 2` 时（self 为 2D 张量 [M, K]），代码使用 `dimData = {0, 1}` 添加两个维度变为 [1, 1, M, K]（4D）。这两种情况的 unsqueeze 维度逻辑是**交换**的。正确逻辑应为：1D 输入需要添加两个维度 `{0, 1}` 变为 [1, 1, K]（3D，对应 BMM 的 [batch, 1, K]），2D 输入需要添加一个维度 `{0}` 变为 [1, M, K]（3D，对应 BMM 的 [batch, M, K]）。
- **触发条件**: 当 self 为 1D 或 2D 张量，mat2 为 >=3D 张量时（例如 `[K] x [B, K, N]` 或 `[M, K] x [B, K, N]`），BatchMatmul 将收到维度不正确的输入，导致计算错误或运行时崩溃。
- **测试方案**:
  ```python
  # Case 1: 1D x 3D
  self = torch.randn(4)            # [K=4]
  mat2 = torch.randn(2, 4, 3)     # [B=2, K=4, N=3]
  out = torch.matmul(self, mat2)   # 期望 [2, 3]，实际可能异常

  # Case 2: 2D x 3D
  self = torch.randn(5, 4)         # [M=5, K=4]
  mat2 = torch.randn(2, 4, 3)     # [B=2, K=4, N=3]
  out = torch.matmul(self, mat2)   # 期望 [2, 5, 3]，实际可能异常
  ```

---

### Bug 2: 空张量处理中 AllocTensor 使用错误的数据类型

- **位置**: 第 254 行，`ProcessEmptyTensor` 函数；第 521 行，`MatMulEmptyTensorGraph::Impl()`
- **类型**: 数据类型错误
- **严重程度**: 中等 (Medium)
- **描述**: `executor->AllocTensor(outShape, self->GetDataType())` 使用了输入张量 `self`（即 `matA`）的数据类型来分配输出张量，而非输出张量 `out`（即 `output`）的数据类型。在类型提升场景下（如 self 为 float16，out 为 float32），分配的输出张量类型与期望不匹配，导致后续数据写入目标 out 时类型不一致。
- **触发条件**: 当输入 self 的数据类型与输出 out 的数据类型不同（例如 self 为 DT_FLOAT16，经类型推导后 out 为 DT_FLOAT），且触发空张量路径（self 或 mat2 的某个维度为 0）。
- **测试方案**:
  ```python
  # self 为 float16，out 为 float32，且 self 或 mat2 含空维度
  self = torch.randn(0, 4, dtype=torch.float16).npu()
  mat2 = torch.randn(4, 3, dtype=torch.float16).npu()
  out = torch.empty(0, 3, dtype=torch.float32).npu()
  # 检查输出张量 dtype 是否正确
  ```

---

### Bug 3: aclnnMatmulWeightNzGetWorkspaceSize 中执行器创建与参数校验顺序不一致

- **位置**: 第 929-933 行，`aclnnMatmulWeightNzGetWorkspaceSize` 函数
- **类型**: 资源管理/逻辑顺序错误
- **严重程度**: 低 (Low)
- **描述**: 在 `aclnnMatmulWeightNzGetWorkspaceSize` 中，先创建 `OpExecutor`（第 929 行），再进行参数校验 `CheckWeightNzParam`（第 933 行）。而对比 `aclnnMatmulGetWorkspaceSize`（第 886-890 行），是先进行参数校验再创建执行器。如果参数校验失败提前返回，虽然 RAII 会清理执行器，但存在不必要的资源分配开销，且与同文件其他 API 的编码风格不一致，增加维护风险。
- **触发条件**: 当传入无效参数（如空指针、不支持的数据类型）时，会先分配执行器资源再释放。
- **测试方案**:
  ```python
  # 传入不支持的数据类型触发参数校验失败
  self = torch.randn(4, 4, dtype=torch.int32).npu()  # 不支持的类型
  mat2 = torch.randn(4, 4, dtype=torch.float16).npu()
  # 调用 aclnnMatmulWeightNzGetWorkspaceSize，观察是否有资源泄漏
  ```

---

### Bug 4: 注释与代码逻辑不匹配

- **位置**: 第 844 行
- **类型**: 注释错误
- **严重程度**: 低 (Low)
- **描述**: 注释写的是 `// dimTensor1 is 1 or 2 && dimTensor2 >= 3`，但实际代码条件是 `dimTensor1 >= 3 && (dimTensor2 == 1 || dimTensor2 == 2)`。注释内容与条件完全相反/交换，容易误导后续开发者维护代码。
- **触发条件**: 开发者阅读代码进行维护或扩展时可能被误导。
- **测试方案**: 代码审查确认注释应改为 `// dimTensor1 >= 3 && dimTensor2 is 1 or 2`。

---

### Bug 5: NZ_K0_VALUE_32 常量命名与实际值严重不符

- **位置**: 第 53 行
- **类型**: 命名错误/可维护性问题
- **严重程度**: 低 (Low)
- **描述**: 常量 `NZ_K0_VALUE_32` 的值为 `8`，命名中的 "32" 指的是 float32 数据类型（32 bit），但极易被误读为该常量的值应该是 32。对比 `NZ_K0_VALUE_16 = 16`（float16 对应 k0=16），这种命名风格不一致。建议改名为 `NZ_K0_VALUE_FP32 = 8` 或 `NZ_K0_VALUE_FOR_FLOAT32 = 8`。
- **触发条件**: 开发者维护代码时可能误以为该值应为 32，导致引入新 bug。
- **测试方案**: 静态代码审查，确认所有使用该常量的地方语义正确。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 733-737 行 | 逻辑错误 | 严重 | Unsqueeze 维度在 1D/2D 情况下交换，导致 BMM 维度错误 |
| 2 | 第 254, 521 行 | 数据类型错误 | 中等 | 空张量输出使用 self 的 dtype 而非 out 的 dtype |
| 3 | 第 929-933 行 | 资源管理 | 低 | 执行器创建在参数校验之前，与其他 API 不一致 |
| 4 | 第 844 行 | 注释错误 | 低 | 注释描述与代码条件完全相反 |
| 5 | 第 53 行 | 命名错误 | 低 | NZ_K0_VALUE_32=8 命名与值不匹配，易误导 |
