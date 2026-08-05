# Ascend NPU 算子代码审查报告 - aclnn_swish.cpp (A88)

## Bug 列表

### Bug 1: CheckDtypeValidBetaToFloat 函数定义但未被调用

- **位置**: 第 42-50 行定义，第 58-68 行 `CheckParams` 函数中缺失调用
- **类型**: 逻辑遗漏 (Missing Validation)
- **严重程度**: 高
- **描述**: `CheckDtypeValidBetaToFloat` 函数用于检查 `betaOptional` 的数据类型是否能安全转换为 `DT_FLOAT`，但在 `CheckParams` 参数校验流程中从未被调用。这意味着当用户传入一个无法转换为 float 的 scalar 类型（如复数类型）时，后续第 110 行 `betaOptional->ToFloat()` 可能产生未定义行为或精度丢失，且不会给出有意义的错误提示。
- **触发条件**: 传入 `betaOptional` 参数的数据类型为不能 cast 到 float 的类型（如 `DT_COMPLEX64`、`DT_COMPLEX128` 等），此时不会报错直接执行，导致计算结果错误或崩溃。
- **测试方案**:
  ```python
  # 构造一个 complex 类型的 aclScalar 作为 betaOptional
  import torch
  import torch_npu
  x = torch.randn(4, 4, dtype=torch.float32).npu()
  beta = torch.tensor(1+2j)  # complex scalar
  # 调用 swish 算子，期望返回 ACLNN_ERR_PARAM_INVALID，实际可能崩溃或静默计算错误
  ```
- **修复建议**: 在 `CheckParams` 中添加对 `betaOptional` 的类型校验调用：
  ```cpp
  CHECK_RET(CheckDtypeValidBetaToFloat(betaOptional), ACLNN_ERR_PARAM_INVALID);
  ```

---

### Bug 2: ReshapeSelfValueGetActivation 使用原始 self 而非 selfContiguous

- **位置**: 第 106 行
- **类型**: 逻辑错误 (Wrong Variable)
- **严重程度**: 中
- **描述**: 第 101 行已将 `self` 做了 Contiguous 操作得到 `selfContiguous`，后续计算应基于连续内存的 tensor 进行。但第 106 行 `ReshapeSelfValueGetActivation` 的第一个参数传入了原始的 `self` 而非 `selfContiguous`。如果该函数内部依赖第一个参数的 stride/offset 等存储信息（而不仅仅是 shape），则会在非连续输入场景下产生错误结果。即使函数仅用 shape 信息，传入原始 tensor 也违反了代码语义一致性，存在潜在风险。
- **触发条件**: 输入 `self` 为非连续 tensor（如 transpose 后的 tensor、slice 后的 tensor），且 `ReshapeSelfValueGetActivation` 内部访问了第一个参数的 stride 或数据指针。
- **测试方案**:
  ```python
  import torch
  import torch_npu
  x = torch.randn(8, 8, dtype=torch.float32).npu().t()  # 非连续 tensor
  out = torch.empty_like(x.contiguous()).npu()
  # 对比 swish 输出与 CPU 参考结果，检查是否一致
  ```
- **修复建议**: 将第 106 行改为使用 `selfContiguous`：
  ```cpp
  auto reshapeSelf = ReshapeSelfValueGetActivation(selfContiguous, dimSize, selfContiguous, uniqueExecutor);
  ```

---

### Bug 3: dimSize 取自原始 self 而非 selfContiguous（一致性问题）

- **位置**: 第 104 行
- **类型**: 代码规范/潜在风险
- **严重程度**: 低
- **描述**: 第 104 行 `dimSize = self->GetViewShape().GetDimNum()` 从原始 `self` 获取维度数。虽然 Contiguous 操作通常不改变 shape，但从代码正确性和一致性角度，后续所有操作均应基于 `selfContiguous`，避免在特殊情况（如 0-dim tensor 的特殊处理）下出现不一致。
- **触发条件**: 一般场景下不会触发实际错误，但若框架内部 Contiguous 对 shape 有特殊处理时可能出现不一致。
- **测试方案**: 单元测试对比 `self->GetViewShape().GetDimNum()` 与 `selfContiguous->GetViewShape().GetDimNum()` 在各种输入下是否始终一致。
- **修复建议**:
  ```cpp
  size_t dimSize = selfContiguous->GetViewShape().GetDimNum();
  ```

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 58-68 行 (CheckParams) | 逻辑遗漏 | 高 | `CheckDtypeValidBetaToFloat` 已定义但未在参数校验中调用，betaOptional 类型未校验 |
| 2 | 第 106 行 | 逻辑错误 | 中 | `ReshapeSelfValueGetActivation` 第一个参数应为 `selfContiguous` 而非 `self` |
| 3 | 第 104 行 | 代码规范 | 低 | `dimSize` 应从 `selfContiguous` 获取以保持一致性 |
