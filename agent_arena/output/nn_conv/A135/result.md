# aclnn_convolution.cpp 代码审查报告

## Bug 1: `All` 函数递归调用错误导致语义不正确

- **位置**: 第 266-273 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `All` 模板函数本意是检查所有参数是否满足条件，但递归调用时错误地调用了 `Any` 而非 `All`。这导致 `All(value, f, a, b, c)` 实际语义变成"第一个满足 AND 剩余中任意一个满足"，而非"所有都满足"。
- **触发条件**: 当 `CHECK_PARAM_ALL_GTE` 或 `CHECK_PARAM_ALL_EQ` 宏被调用且参数列表有 3 个以上比较值时，第一个元素通过检查后，后续只要有任意一个通过就不会报错，导致非法参数被放行。
- **测试方案**: 构造 input shape 为 `[0, -1, 4, 4]`（N>=0 但 C<0），调用卷积，验证 `CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, ...)` 能否正确拦截 C 为负数的情况。

## Bug 2: 永远为假的条件判断（不可达的校验逻辑）

- **位置**: 第 1351 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 条件 `if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))` 永远为 `false`，因为同一变量不可能同时 `< 0` 且 `== 0`。正确逻辑应为 `if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0))`。
- **触发条件**: 在 transposed 模式下 (ASCEND910_95)，当 weight 某空间维度为负数或为 0 但对应 output 维度非 0 时，该校验完全失效，非法参数不会被拦截，可能导致后续计算错误或内存越界。
- **测试方案**: 构造 transposed conv 场景，weight shape 中某空间维度设为 -1 或 0（output 对应维度非0），验证是否能触发 `ACLNN_ERR_PARAM_INVALID` 错误。

## Bug 3: workspace 大小被硬编码为 0

- **位置**: 第 4402 行
- **类型**: 功能错误
- **严重程度**: 高
- **描述**: 在 `aclnnConvolutionGetWorkspaceSize` 函数中，`*workspaceSize = 0;` 直接将 workspace 大小写为 0，而未调用 `uniqueExecutor.get()->GetWorkspaceSize()` 获取实际所需大小。对比 `aclnnConvTbcGetWorkspaceSize`（第 4476 行）正确地调用了 `GetWorkspaceSize()`。
- **触发条件**: 任何 aclnnConvolution 调用场景下，返回的 workspaceSize 始终为 0，导致上层分配 0 字节 workspace，在执行阶段若算子实际需要 workspace 则可能出现内存越界或计算错误。
- **测试方案**: 调用 `aclnnConvolutionGetWorkspaceSize`，检查返回的 `workspaceSize` 是否为 0；对比调用 `aclnnConvTbc` 同等规模输入时的 workspaceSize；验证执行阶段是否出现非法内存访问。

## Bug 4: 变量命名与值语义不匹配

- **位置**: 第 67 行
- **类型**: 命名错误/语义错误
- **严重程度**: 中
- **描述**: `static const std::string REFLECTION_MODE = "constant";` 变量名为 `REFLECTION_MODE`（反射模式），但实际值为 `"constant"`（常量填充模式）。在第 2311 行作为 PadV3 的 mode 参数使用时，传入的是 "constant" 模式，功能上此处需要的是常量填充，但变量名极易误导开发者。若后续有人依赖变量名语义进行修改或使用，会引入错误。
- **触发条件**: 开发者在其他地方复用 `REFLECTION_MODE` 期望获得反射填充模式时，实际传递的是常量填充模式。
- **测试方案**: 代码审查确认 PadV3 调用处的意图；搜索所有使用 `REFLECTION_MODE` 的位置确认是否都期望 "constant" 值。

## Bug 5: `std::map` 按值传递导致性能浪费

- **位置**: 第 130 行、第 192 行
- **类型**: 性能问题
- **严重程度**: 低
- **描述**: `ConvL0Warper` 和 `L0FuncWarperByOpType` 函数的第一个参数 `std::map<std::string, L0FUNCTION> l0Functions` 按值传递，每次调用都会拷贝整个 map。应改为 `const std::map<std::string, L0FUNCTION>& l0Functions`。
- **触发条件**: 每次卷积调用都会触发 map 拷贝，在高频调用场景下浪费 CPU 时间和内存。
- **测试方案**: 使用性能分析工具对比按值传递和按引用传递的耗时差异；或通过代码审查确认。

## Bug 6: `isNotDMA` 函数中 outputW 初始赋值取错维度

- **位置**: 第 2493 行
- **类型**: 逻辑错误
- **严重程度**: 低
- **描述**: `int64_t outputW = (int64_t)output->GetViewShape().GetDim(2);` 对于 NCHW 格式的 4D tensor，dim(2) 是 H 而不是 W。虽然后续有 `if (outputSize == CONV_2D_DIM_SIZE)` 会修正为 dim(3)，但当 output 不是 4D 时（防御性场景），outputW 实际存储的是 H 维度值，可能导致后续 L1 切分计算错误。
- **触发条件**: 当 output 的维度数不等于 4 时（虽然在当前调用链中不太可能），outputW 值不正确。
- **测试方案**: 单元测试覆盖 `isNotDMA` 函数，传入非 4D output tensor 观察行为是否正确。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 266-273 | 逻辑错误 | 高 | `All` 函数递归误调 `Any`，无法正确校验"所有参数满足条件" |
| 2 | 1351 | 逻辑错误 | 高 | `< 0 && == 0` 永远为 false，校验逻辑失效 |
| 3 | 4402 | 功能错误 | 高 | workspaceSize 硬编码为 0，未获取实际所需值 |
| 4 | 67 | 命名/语义错误 | 中 | `REFLECTION_MODE` 实际值为 "constant"，名称与值不匹配 |
| 5 | 130, 192 | 性能问题 | 低 | std::map 按值传递造成不必要的拷贝开销 |
| 6 | 2493 | 逻辑错误 | 低 | outputW 默认取 dim(2) 即 H 维度，非 4D 场景下值错误 |
