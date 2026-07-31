# Ascend 910B 算子代码审查报告

文件: `aclnn_convolution.cpp`

---

## Bug 1: `All` 模板函数递归调用错误（调用了 `Any` 而非 `All`）

- **位置**: 第 266-273 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `All` 函数旨在检查所有参数是否满足条件，但在递归调用时错误地调用了 `Any` 而非 `All`。当变长参数列表有 3 个或以上元素时，第一个元素检查通过后，剩余元素只需满足任意一个即返回 true，违背了"全部满足"的语义。
- **问题代码**:
  ```cpp
  template <typename T, typename Func, typename... LIST>
  static inline bool All(T value, Func f, T compare, LIST... list)
  {
      bool result = f(value, compare);
      if (result) {
          return Any(value, f, list...);  // BUG: 应为 All(value, f, list...)
      }
      return false;
  }
  ```
- **触发条件**: 当 `CHECK_PARAM_ALL_GTE` 或 `CHECK_PARAM_ALL_EQ` 宏被调用且变长参数列表有 3 个或以上参数时触发。例如第 1581 行 `CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, weightShapeN, weightShapeC)` — 若 `inputShapeN >= 0` 且 `inputShapeC >= 0`，但 `weightShapeN < 0` 而 `weightShapeC >= 0`，检查会错误通过。
- **测试方案**: 构造 input shape N>=0, C>=0, weight shape N<0 (非法负值), C>=0 的场景，调用卷积接口验证 `CheckShape` 是否能正确拦截非法 weight N 值。

---

## Bug 2: `CheckEmptyTensorTransposed` 中逻辑条件永假

- **位置**: 第 1351 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 条件 `if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))` 永远为 false，因为 `weightShape[i] < 0` 和 `weightShape[i] == 0` 不可能同时为 true。正确意图应为使用 `||` 连接两个子条件，即 `if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0))`。
- **问题代码**:
  ```cpp
  if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0)) {
  ```
- **触发条件**: 在 transposed 模式且平台为 ASCEND910_95 时，weight 的空间维度为负值或者 weight 空间维度为 0 但 output 对应维度非 0 的非法输入将无法被检测到。
- **测试方案**: 在 910_95 平台上，构造 transposed convolution，设 weight shape 中某空间维度为 -1 或设 weight 某空间维度=0 而 output 对应维度>0，验证是否报错。

---

## Bug 3: C04 分支 weight padding 使用错误的填充模式 "reflect"

- **位置**: 第 2311 行
- **类型**: 语义/计算错误
- **严重程度**: 高
- **描述**: 在 C04 权重预处理中，需要将 weight 的 C 维度从 <4 零填充至 4。但使用了 `op::REFLECTION_MODE`（即 `"reflect"` 反射填充），这会镜像复制已有通道数据而非填充零值，导致 weight 数据被错误修改，卷积计算结果不正确。
- **问题代码**:
  ```cpp
  weight = l0op::PadV3(weight, paddingTensor, constantValues, op::REFLECTION_MODE, true, executor);
  ```
- **触发条件**: 当输入满足 C04 分支条件（groups=1, Cin<=4, FP16 dtype, 910B, 非 DMA 等）且 weight 的 C 维度不等于 4 时触发。例如 Cin=3 的 RGB 输入卷积在满足 C04 条件时进入该分支。
- **测试方案**: 构造 input shape [1, 3, 224, 224], weight shape [64, 3, 3, 3], dtype=FP16 在 910B 上运行卷积，对比结果与标准卷积输出的精度差异。

---

## Bug 4: `ConvL0Warper` 中 map 按值传递导致性能损失

- **位置**: 第 130 行
- **类型**: 性能问题
- **严重程度**: 低
- **描述**: `ConvL0Warper` 函数的第一个参数 `std::map<std::string, L0FUNCTION> l0Functions` 是按值传递的，每次调用都会完整拷贝整个 map。应改为 `const std::map<std::string, L0FUNCTION>& l0Functions` 以避免不必要的拷贝。同样问题存在于第 192 行的 `L0FuncWarperByOpType`。
- **触发条件**: 每次调用卷积算子时均触发。
- **测试方案**: 性能基准测试，对比修改前后的算子调度耗时。

---

## Bug 5: `CanSwitchC04InBF16Scene` 返回值逻辑反转

- **位置**: 第 2553-2561 行
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: `CanSwitchC04InBF16Scene` 函数在 BF16 + (910_93 或 910B) 时返回 `true` 表示"可以切换到 C04"。然而该函数的注释和上下文说明 C04 仅在 FP16 场景需要特定芯片支持（`IsCubeSupportFp32`），BF16 场景直接走 C04 时实际需要验证硬件是否支持 BF16+C04 组合。如果 BF16 的 C04 在实际硬件上不支持而函数返回 true，会导致运行时错误。此处需确认硬件兼容性，目前存在风险。
- **触发条件**: BF16 输入，910B/910_93 平台，且满足 C04 其他条件时。
- **测试方案**: 在 910B 上使用 BF16 输入、Cin<=4 的卷积验证 C04 路径是否正确执行。

---

## Bug 6: `aclnnConvDepthwise2dGetWorkspaceSize` 中 groups 使用初始值 1 进行 check

- **位置**: 第 4501-4509 行
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: 在 `aclnnConvDepthwise2dGetWorkspaceSize` 中，`groups` 在第 4501 行被初始化为 1，然后用这个值构造 `ConvParams` 并执行 `CheckConvDepthwise2dParams`。但实际 groups 应等于输入的 channel 数（在第 4518 行才被更新）。这意味着 check 阶段使用了错误的 groups 值，可能导致某些合法输入被错误拦截或非法输入被放过。
- **触发条件**: Depthwise2d 卷积中，groups 应等于输入 channel 数，而 check 使用 groups=1 进行校验。
- **测试方案**: 构造 depthwise2d 卷积输入，设 inChannel>1，验证 check 逻辑中 groups 相关校验是否正确（如 weight.C * groups == inChannel）。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 270 | 逻辑错误 | 高 | `All` 递归误调 `Any`，多参数校验失效 |
| 2 | 1351 | 逻辑错误 | 高 | `&&` 导致条件永假，weight 负值校验失效 |
| 3 | 2311 | 语义错误 | 高 | C04 weight padding 使用 reflect 模式而非 constant |
| 4 | 130, 192 | 性能问题 | 低 | map 按值传递导致不必要拷贝 |
| 5 | 2553-2561 | 逻辑风险 | 中 | BF16 C04 分支硬件兼容性未充分验证 |
| 6 | 4501-4509 | 逻辑错误 | 中 | Depthwise2d 中 groups=1 进行 check，与实际不符 |
