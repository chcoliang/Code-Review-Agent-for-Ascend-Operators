# Ascend NPU 算子代码审查报告 - aclnn_swish.cpp (A84)

## Bug 列表

### Bug 1: ReshapeSelfValueGetActivation 使用了非 Contiguous 的原始 Tensor

- **位置**: 第 107 行
- **类型**: 逻辑错误 / 参数传递错误
- **严重程度**: 高
- **描述**: 在第 102 行已经通过 `l0op::Contiguous(self, ...)` 得到了连续内存的 `selfContiguous`，但第 107 行调用 `ReshapeSelfValueGetActivation` 时第一个参数传入的是原始的 `self` 而非 `selfContiguous`。如果原始 tensor 是非连续的(如转置、切片后的 tensor)，对其进行 Reshape 操作会导致数据布局错误或计算结果错误。
- **触发条件**: 当输入 tensor `self` 为非连续存储(例如经过 transpose、slice、permute 等操作后的 tensor)，且维度数超过 `MAX_SUPPORT_DIMS_NUMS` 需要 reshape 时触发。
- **修复建议**: 将第 107 行的 `self` 改为 `selfContiguous`:
  ```cpp
  auto reshapeSelf = ReshapeSelfValueGetActivation(selfContiguous, dimSize, selfContiguous, uniqueExecutor);
  ```
- **测试方案**: 构造一个经过 permute 或 transpose 后非连续的高维(>8维) tensor，调用 aclnnSwish，比较输出与 PyTorch `x * sigmoid(beta * x)` 的参考结果，验证数值是否一致。

---

### Bug 2: dimSize 取自原始 self 而非 selfContiguous

- **位置**: 第 105 行
- **类型**: 逻辑错误
- **严重程度**: 低
- **描述**: `dimSize` 从 `self->GetViewShape().GetDimNum()` 获取。虽然 Contiguous 操作通常不改变 shape 维度数，但从代码一致性和健壮性角度，后续所有操作都应基于 `selfContiguous` 进行，以避免在某些特殊 format 转换场景下 view shape 不一致的潜在问题。
- **触发条件**: 特殊内部 format 转换场景下，Contiguous 后 view shape 维度数与原始不同时。
- **修复建议**: 改为 `size_t dimSize = selfContiguous->GetViewShape().GetDimNum();`
- **测试方案**: 使用不同 format(如 NZ format) 的输入 tensor 进行测试，验证维度信息获取的正确性。

---

### Bug 3: ASCEND910B 数据类型支持列表缺少 BF16

- **位置**: 第 39-40 行
- **类型**: 功能缺陷 / 数据类型支持不完整
- **严重程度**: 中
- **描述**: Ascend 910B 平台硬件支持 BF16 (BFloat16) 数据类型的计算，Swish 作为常见激活函数在大模型训练中广泛使用 BF16 精度。但 `ASCEND910B_DTYPE_SUPPORT_LIST` 中仅包含 `DT_FLOAT` 和 `DT_FLOAT16`，未包含 `DT_BF16`，导致 BF16 输入会被拒绝或需要额外 cast，降低性能。
- **触发条件**: 在 Ascend 910B 平台上，使用 BF16 数据类型的 tensor 调用 aclnnSwish 时，会返回 `ACLNN_ERR_PARAM_INVALID` 错误。
- **修复建议**: 在 910B 支持列表中添加 BF16:
  ```cpp
  static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
      op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
  ```
- **测试方案**: 在 910B 平台上构造 BF16 类型输入 tensor，验证 Swish 算子能正常执行并输出正确结果。

---

### Bug 4: reshapeLongTensor 函数 valuePerm 默认值为 nullptr 存在空指针风险

- **位置**: 第 71-79 行
- **类型**: 防御性编程缺陷
- **严重程度**: 低
- **描述**: 函数 `reshapeLongTensor` 的参数 `valuePerm` 默认值为 `nullptr`。如果未来有人在不传入 `valuePerm` 的情况下调用该函数，且 tensor 需要 reshape（即不满足第 74 行的 early return 条件），则 `l0op::Reshape(x, nullptr, executor)` 会导致未定义行为或空指针解引用。当前调用点(第 119 行)传入了有效的 `shapeOriDetial`，但函数接口设计存在隐患。
- **触发条件**: 如果有新的调用点不传入 valuePerm 参数，且 tensor 维度不满足 early-return 条件，就会触发空指针问题。
- **修复建议**: 移除默认参数，强制调用方提供 valuePerm；或者在函数内添加 nullptr 检查:
  ```cpp
  if (valuePerm == nullptr) {
      OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "valuePerm is nullptr in reshapeLongTensor");
      return nullptr;
  }
  ```
- **测试方案**: 代码静态分析验证所有调用点均传入有效 valuePerm；单元测试覆盖高维 tensor reshape 路径。

---

### Bug 5: GetTensorShapeActivation 无条件调用造成冗余计算且返回值未做空指针检查

- **位置**: 第 106 行
- **类型**: 缺少空指针检查 / 性能问题
- **严重程度**: 中
- **描述**: `GetTensorShapeActivation` 在所有情况下都被调用，但其返回值 `shapeOriDetial` 仅在 `dimSize > MAX_SUPPORT_DIMS_NUMS` 时使用(第 119 行)。更关键的是，该返回值未进行空指针检查，如果函数因内存分配失败等原因返回 nullptr，后续传入 `reshapeLongTensor` 再传入 `l0op::Reshape` 时将导致未定义行为。
- **触发条件**: 当 `GetTensorShapeActivation` 因内部错误返回 nullptr，且 `dimSize > MAX_SUPPORT_DIMS_NUMS` 时触发。
- **修复建议**:
  ```cpp
  if (dimSize > MAX_SUPPORT_DIMS_NUMS) {
      auto shapeOriDetial = GetTensorShapeActivation(selfContiguous, uniqueExecutor.get());
      CHECK_RET(shapeOriDetial != nullptr, ACLNN_ERR_INNER_NULLPTR);
      reshapeSwishOut = reshapeLongTensor(swishOut, uniqueExecutor.get(), dimSize, shapeOriDetial);
  }
  ```
- **测试方案**: 模拟内存不足场景，验证 `GetTensorShapeActivation` 返回 nullptr 时算子能安全退出并返回错误码。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 107 行 | 逻辑错误 | 高 | Reshape 使用非 Contiguous 的原始 tensor，非连续输入时计算结果错误 |
| 2 | 第 105 行 | 逻辑错误 | 低 | dimSize 应从 selfContiguous 获取，保持一致性 |
| 3 | 第 39-40 行 | 功能缺陷 | 中 | 910B 平台未支持 BF16 数据类型 |
| 4 | 第 71-79 行 | 防御性编程缺陷 | 低 | valuePerm 默认 nullptr 存在潜在空指针风险 |
| 5 | 第 106 行 | 空指针检查缺失 | 中 | GetTensorShapeActivation 返回值未做空指针校验 |
