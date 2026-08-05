# aclnn_leaky_relu.cpp 代码审查报告

## Bug 列表

### Bug 1: 缺少 Contiguous 转换调用

- **位置**: 第 100-101 行
- **类型**: 逻辑缺陷 / 功能缺失
- **严重程度**: 严重 (Critical)
- **描述**: 代码注释明确说明"将输入tensor转换成连续的tensor"，并且已包含头文件 `"aclnn_kernels/contiguous.h"`，但实际代码仅执行了 `auto selfContiguous = self;`（指针赋值），并未调用 `l0op::Contiguous()` 进行真正的连续化处理。当输入 tensor 的内存布局不连续（如经过 slice、transpose、permute 等操作后）时，后续 `LeakyRelu` kernel 会读取错误的数据，导致计算结果错误或内存越界访问。
- **触发条件**: 输入 self tensor 为非连续内存布局（non-contiguous），例如通过 `tensor.transpose()` 或 `tensor[::2]` 等操作获得的 view tensor。
- **修复建议**: 应改为：
  ```cpp
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  ```
- **测试方案**:
  1. 构造一个转置后的非连续 tensor 作为输入，执行 LeakyReLU，对比与 CPU 参考实现的结果。
  2. 构造 stride 不规则的 view tensor（如 `x[::2, ::3]`），验证输出正确性。
  3. 使用内存检测工具检查是否有越界读取。

### Bug 2: negativeSlope 精度丢失

- **位置**: 第 104 行
- **类型**: 精度缺陷
- **严重程度**: 中等 (Medium)
- **描述**: `negativeSlope->ToFloat()` 将 scalar 强制转换为 32 位 float。当输入 tensor 数据类型为 `DT_DOUBLE`（双精度浮点）时，negativeSlope 的精度会从 64 位截断到 32 位，导致计算结果精度下降。算子支持列表中明确包含 `DT_DOUBLE` 类型。
- **触发条件**: 输入 tensor 类型为 `DT_DOUBLE`，且 negativeSlope 值需要超过 float 精度才能正确表示（如 `0.123456789012345`）。
- **修复建议**: 根据输入 tensor 的数据类型选择合适的转换精度，当输入为 DT_DOUBLE 时使用 `ToDouble()`。
- **测试方案**:
  1. 使用 DT_DOUBLE 类型输入，设置 negativeSlope 为高精度值（如 1e-15 级别的差异），对比输出与 CPU 双精度参考结果。
  2. 验证 float 和 double 场景下 negativeSlope 精度保持一致。

### Bug 3: 缺少输出 tensor 数据类型校验

- **位置**: 第 54-58 行（`CheckDtypeValid` 函数）
- **类型**: 参数校验不完整
- **严重程度**: 中等 (Medium)
- **描述**: `CheckDtypeValid` 仅校验了输入 `self` 的数据类型是否在支持列表中，但未校验输出 `out` 的数据类型是否合法。如果用户传入不支持类型的输出 tensor（如 INT8），后续的 `Cast` 操作可能失败或产生未定义行为。
- **触发条件**: 用户创建一个数据类型为不支持类型（如 DT_INT8、DT_INT32）的输出 tensor 传入 API。
- **修复建议**: 在 `CheckDtypeValid` 中增加对 `out` 数据类型的校验，或至少校验输出类型属于支持的浮点类型之一。
- **测试方案**:
  1. 传入 DT_INT8 类型的输出 tensor，验证 API 返回合适的错误码而非崩溃。
  2. 传入 DT_BOOL 类型输出 tensor，检查是否能正确拦截。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 100-101 行 | 逻辑缺陷 | 严重 | 缺少 Contiguous 调用，非连续 tensor 输入时计算结果错误 |
| 2 | 第 104 行 | 精度缺陷 | 中等 | negativeSlope 强制转 float，DT_DOUBLE 场景精度丢失 |
| 3 | 第 54-58 行 | 校验不完整 | 中等 | 未校验输出 tensor 的数据类型合法性 |
