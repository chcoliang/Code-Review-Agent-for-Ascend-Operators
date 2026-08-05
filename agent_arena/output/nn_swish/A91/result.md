# Ascend NPU 算子代码审查报告 - aclnn_swish.cpp (A91)

## Bug 列表

### Bug 1: 缺少 Contiguous 操作导致非连续张量计算错误

- **位置**: 第 102-103 行
- **类型**: 逻辑缺陷 / 功能缺失
- **严重程度**: 高
- **描述**: 代码中 `selfContiguous` 直接赋值为 `self`，并未实际调用 `l0op::Contiguous` 将输入张量转为连续存储格式。尽管头文件 `aclnn_kernels/contiguous.h` 已被包含（第15行），但从未使用。变量命名 `selfContiguous` 暗示此处应执行连续化操作，但实际只是指针拷贝。当输入张量 `self` 的存储非连续（如经过 transpose、slice 等操作后）时，后续的 Swish 计算将在非连续内存上操作，产生错误结果或访问越界。
  
  正确写法应类似:
  ```cpp
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  ```

- **触发条件**: 输入张量 `self` 为非连续存储格式（如经过 `transpose`、`permute`、`slice`、`narrow` 等操作后未调用 `contiguous()` 的张量）。
- **测试方案**:
  1. 创建一个连续张量 A，对其执行 `transpose` 操作得到非连续张量 B。
  2. 将 B 作为 `self` 输入调用 `aclnnSwish`。
  3. 对比输出结果与参考实现（先 contiguous 再 swish）的结果，验证数值是否一致。
  4. 预期：当前实现结果不正确或触发内存访问异常。

---

### Bug 2: reshapeLongTensor 函数条件判断逻辑存在缺陷

- **位置**: 第 73-76 行
- **类型**: 逻辑缺陷
- **严重程度**: 中
- **描述**: `reshapeLongTensor` 函数的早返回条件为 `originalDimSize == dimSize && dimSize <= MAX_SUPPORT_DIMS_NUMS`。该函数在第119行被调用时，`originalDimSize` 为原始张量的维度数（已知 > MAX_SUPPORT_DIMS_NUMS），而 `dimSize` 为 `swishOut` 的实际维度数。此处逻辑依赖于 Swish 输出的维度数一定与输入 reshape 后的维度数相同（即 <= MAX_SUPPORT_DIMS_NUMS）。如果某些边界情况下 `swishOut` 的维度数恰好等于 `originalDimSize`（虽然不太可能），则会错误地跳过 reshape 回原始形状的操作。此外，当 `valuePerm` 为 nullptr（默认值）时调用 `l0op::Reshape` 可能导致未定义行为。

- **触发条件**: 极端情况下 Swish 算子输出的维度数与原始输入维度数相同；或在非第119行的调用路径中 `valuePerm` 为 nullptr。
- **测试方案**:
  1. 构造维度数超过 `MAX_SUPPORT_DIMS_NUMS` 的输入张量。
  2. 执行 `aclnnSwish` 并验证输出张量的 shape 是否与输入一致。
  3. 检查 reshape 操作是否正确执行。

---

### Bug 3: ReshapeSelfValueGetActivation 参数传递可能不一致

- **位置**: 第 107 行
- **类型**: 逻辑缺陷
- **严重程度**: 中
- **描述**: `ReshapeSelfValueGetActivation` 函数同时接收 `self` 和 `selfContiguous` 作为参数，但由于 Bug 1 的存在，这两个参数实际上是同一个指针。该函数的设计意图应该是对连续化后的张量进行 reshape 操作，但由于 `selfContiguous` 未经连续化处理，传入的是原始非连续张量，可能导致 reshape 结果不正确。此外，`uniqueExecutor` 以值方式传入（非 `.get()`），需确认其接口是否支持该用法。

- **触发条件**: 输入张量为非连续存储且维度数超过 MAX_SUPPORT_DIMS_NUMS。
- **测试方案**:
  1. 创建高维度非连续张量作为输入。
  2. 验证 reshape 后的中间张量数据布局是否正确。
  3. 对比最终输出与 CPU 参考实现的结果。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 102-103 行 | 逻辑缺陷/功能缺失 | 高 | 缺少 Contiguous 调用，非连续张量输入时计算结果错误 |
| 2 | 第 73-76 行 | 逻辑缺陷 | 中 | reshapeLongTensor 条件判断逻辑脆弱，边界情况可能跳过必要的 reshape |
| 3 | 第 107 行 | 逻辑缺陷 | 中 | ReshapeSelfValueGetActivation 参数因 Bug 1 导致传入未连续化张量 |

## 总结

本文件最核心的问题是 **缺少对输入张量的 Contiguous 操作**（Bug 1），这是一个高严重度的功能缺陷，会导致所有非连续输入张量的计算结果错误。Bug 2 和 Bug 3 属于中等严重度的逻辑问题，在特定维度条件下可能触发。建议优先修复 Bug 1，在第 102 行添加 `l0op::Contiguous` 调用。
