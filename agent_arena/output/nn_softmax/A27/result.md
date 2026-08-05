# Ascend NPU 算子代码审查报告 - aclnn_softmax.cpp (A27)

## Bug 列表

### Bug 1: 空指针检查失败时返回成功状态码

- **位置**: 第 90 行
- **类型**: 逻辑错误 / 错误码误用
- **严重程度**: 严重 (Critical)
- **描述**: `CHECK_RET(CheckNotNull(self, out), ACLNN_SUCCESS);` 当 `CheckNotNull` 返回 `false`（即检测到空指针）时，`CHECK_RET` 宏会返回第二个参数作为错误码。此处第二个参数为 `ACLNN_SUCCESS`，意味着空指针情况下函数返回成功，调用方无法感知错误，后续可能导致空指针解引用崩溃。应改为 `ACLNN_ERR_PARAM_INVALID`。
- **触发条件**: 调用 `aclnnSoftmaxGetWorkspaceSize` 时传入 `self = nullptr` 或 `out = nullptr`。
- **测试方案**: 
  ```cpp
  // 传入空指针，验证返回值应为错误码而非ACLNN_SUCCESS
  aclnnStatus ret = aclnnSoftmaxGetWorkspaceSize(nullptr, 0, out, &wsSize, &executor);
  ASSERT_NE(ret, ACLNN_SUCCESS);
  ```

### Bug 2: 注释与实际调用算子不符（SoftmaxGrad vs SoftmaxV2）

- **位置**: 第 132 行
- **类型**: 注释错误
- **严重程度**: 低 (Low)
- **描述**: 注释写的是 `"调用SoftmaxGrad算子kernel"`，但实际调用的是 `l0op::SoftmaxV2`（前向算子）。这是前向 Softmax 实现，不是梯度算子。错误注释会误导代码维护者，增加后续维护风险。
- **触发条件**: 代码审查/维护时产生误解。
- **测试方案**: 代码审查检查，确认注释与实现一致。修正注释为 `"调用SoftmaxV2算子kernel"`。

### Bug 3: 输出 tensor 维度上限未检查

- **位置**: 第 80-85 行 (`CheckShape` 函数)
- **类型**: 校验遗漏
- **严重程度**: 中等 (Medium)
- **描述**: `CheckShape` 函数仅对输入 `self` 进行 `OP_CHECK_MAX_DIM` 检查（不超过 8 维），但未对输出 `out` 进行同样的维度上限检查。虽然第 83 行 `OP_CHECK_SHAPE_NOT_EQUAL` 会校验 self 和 out 形状一致，间接保证了 out 的维度。但如果 shape 相等检查存在边界情况（如 0 维 tensor），可能遗漏 out 的维度异常。防御性编程应对 out 也做维度检查。
- **触发条件**: 当框架层面构造了维度数超过 8 的 out tensor，且 shape 校验未能拦截时。
- **测试方案**:
  ```cpp
  // 构造超过8维的out tensor，验证是否被正确拦截
  auto out = CreateTensor({1,1,1,1,1,1,1,1,1}, DT_FLOAT); // 9维
  auto self = CreateTensor({1,1,1,1,1,1,1,1,1}, DT_FLOAT);
  aclnnStatus ret = aclnnSoftmaxGetWorkspaceSize(self, 0, out, &wsSize, &executor);
  ASSERT_NE(ret, ACLNN_SUCCESS);
  ```

### Bug 4: 空 tensor 提前返回时跳过输出 tensor 有效性校验

- **位置**: 第 92-95 行
- **类型**: 校验顺序不当
- **严重程度**: 低 (Low)
- **描述**: 当 `self->IsEmpty()` 为 true 时，函数在第 94 行直接返回 `ACLNN_SUCCESS`，跳过了后续对 `out` 的数据类型检查（第 97 行）和形状检查（第 103 行）。如果 `out` 的 dtype 或 shape 不合法，调用者得到成功返回却持有非法的 out tensor，可能在后续使用中引发问题。
- **触发条件**: 传入一个空的 self tensor，同时 out 的 dtype 或 shape 非法。
- **测试方案**:
  ```cpp
  // 空self + 非法out dtype
  auto self = CreateEmptyTensor(DT_FLOAT);
  auto out = CreateEmptyTensor(DT_INT32); // 不支持的dtype
  aclnnStatus ret = aclnnSoftmaxGetWorkspaceSize(self, 0, out, &wsSize, &executor);
  // 期望返回错误或至少不影响后续流程
  ```

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 90 行 | 逻辑错误/错误码误用 | 严重 | 空指针检查失败时返回 `ACLNN_SUCCESS` 而非错误码 |
| 2 | 第 132 行 | 注释错误 | 低 | 注释写 "SoftmaxGrad" 但实际调用 SoftmaxV2 前向算子 |
| 3 | 第 80-85 行 | 校验遗漏 | 中等 | 仅检查 self 维度上限，未检查 out 维度上限 |
| 4 | 第 92-95 行 | 校验顺序不当 | 低 | 空 tensor 提前返回跳过 out 有效性校验 |
