# Ascend NPU aclnn_convolution.cpp 代码审查报告

## Bug 列表

### Bug 1: `All` 模板函数递归调用错误 - 调用 `Any` 而非 `All`

- **位置**: 第 266-273 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `All` 函数的语义是"参数需要满足所有参数列表判断"，但在递归时调用了 `Any` 而非 `All`。这导致 `All` 实际上只检查第一个值满足条件后，后续值只需任一满足即可，与"全部满足"的语义不符。
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
- **触发条件**: 当使用 `CHECK_PARAM_ALL_EQ`、`CHECK_PARAM_ALL_GTE`、`CHECK_PARAM_LT_ALL`、`CHECK_PARAM_GT_ALL` 等宏进行超过2个参数校验时，第二个参数之后只需任意一个满足条件即通过校验，本应拒绝的非法参数可能被放行。
- **测试方案**: 构造 input shape 为 `[1, -1, 3, 3]` 和 weight shape 为 `[1, 1, 3, 3]`，在 `CheckShape` 中调用 `CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, weightShapeN, weightShapeC)`，验证 `inputShapeC = -1` 时是否仍能通过校验（因为只要任一后续参数 >= 0 即可误判为通过）。

---

### Bug 2: `CheckEmptyTensorTransposed` 中 weight 维度校验逻辑恒为 false

- **位置**: 第 1351 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 条件 `if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))` 恒为 `false`，因为 `weightShape[i] < 0` 和 `weightShape[i] == 0` 互斥，不可能同时成立。正确逻辑应为 `||`（或）连接。
```cpp
// 错误代码:
if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))
// 正确应为:
if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0))
```
- **触发条件**: 在 ASCEND910_95 平台上，transposed 模式下 weight 的某个维度为负数或为0（且对应 output 维度非0）时，校验不会报错，非法数据直接进入计算流程，可能导致硬件异常或结果错误。
- **测试方案**: 在 ASCEND910_95 上执行 transposed convolution，设置 weight shape 为 `[3, 3, -1, 3]`（包含负数空间维度），验证是否能正确报出 `ACLNN_ERR_PARAM_INVALID` 错误。

---

### Bug 3: `REFLECTION_MODE` 常量命名与实际值不匹配

- **位置**: 第 67 行，使用位置第 2311 行
- **类型**: 语义错误 / 命名错误
- **严重程度**: 中
- **描述**: 常量名为 `REFLECTION_MODE`（反射填充模式），但值为 `"constant"`（常量填充模式）。在 `CommonPreProcessC04` 的 PadV3 调用中使用该变量，导致代码含义混乱，且如果后续维护者依据名称修改值为 `"reflect"`，会导致C04分支padding行为错误。
```cpp
static const std::string REFLECTION_MODE = "constant"; // 名称暗示反射，值为常量
// 使用处:
weight = l0op::PadV3(weight, paddingTensor, constantValues, op::REFLECTION_MODE, true, executor);
```
- **触发条件**: 当进入 C04 分支且 weight 的 C 维度不等于 4 时，PadV3 使用此模式进行 padding。当前虽然"constant"是正确的行为（用0填充），但命名错误会误导后续开发。
- **测试方案**: 代码审查确认；可通过将常量名改为 `CONSTANT_MODE` 或 `PAD_CONSTANT_MODE` 验证功能不受影响。

---

### Bug 4: `ConvL0Warper` 和 `L0FuncWarperByOpType` 按值传递 `std::map`

- **位置**: 第 130 行、第 191 行
- **类型**: 性能缺陷
- **严重程度**: 中
- **描述**: `ConvL0Warper` 和 `L0FuncWarperByOpType` 函数的第一个参数 `std::map<std::string, L0FUNCTION> l0Functions` 按值传递，导致每次调用都完整拷贝整个 map。在卷积算子的热路径上，这会带来不必要的内存分配和拷贝开销。
```cpp
static const aclTensor* ConvL0Warper(
    std::map<std::string, L0FUNCTION> l0Functions, ...)  // 应为 const std::map<...>& l0Functions
```
- **触发条件**: 每次执行卷积算子时必然触发，在高频调用场景下产生显著性能退化。
- **测试方案**: 对比修改前后的性能基准测试；也可通过 profiling 工具观察到额外的 heap allocation。

---

### Bug 5: `isNotDMA` 函数中 `outputW` 初始赋值使用错误维度索引

- **位置**: 第 2493-2496 行
- **类型**: 代码质量 / 潜在逻辑错误
- **严重程度**: 低
- **描述**: `outputW` 首先被赋值为 `output->GetViewShape().GetDim(2)`（对于 NCHW 格式这是 H 而非 W），然后在 `outputSize == CONV_2D_DIM_SIZE` 时修正为 `GetDim(3)`。虽然当前调用链保证 output 为 4D（所以总会修正），但如果函数被复用于其他维度场景将产生错误。
```cpp
int64_t outputW = (int64_t)output->GetViewShape().GetDim(2);  // 对4D来说这是H
if (outputSize == CONV_2D_DIM_SIZE) {
    outputW = static_cast<int64_t>(output->GetViewShape().GetDim(3)); // 修正为W
}
```
- **触发条件**: 当前由于 `isNotDMA` 仅在2D卷积场景调用，总会进入修正分支。但若代码重构或复用可能暴露问题。
- **测试方案**: 直接将初始赋值改为 `GetDim(3)` 并删除条件判断，验证2D卷积 C04 分支功能不变。

---

### Bug 6: `ConvTransposed2dImpl::PostProcess` 中 SwapHW 后 ViewCopy 目标可能 shape 不匹配

- **位置**: 第 3963-3967 行
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: 在 `PostProcess` 中，`CommonPostProcess` 输出后如果 `ConvTransposed2dSwitchHW` 为 true，对 `convOut` 进行了 H/W 交换。但 `CommonPostProcess` 已经将 `convOut` 转为 output 的 dtype/format，此时再 swap H/W 可能导致与 `output` tensor 的 shape 不一致，`ViewCopy` 可能失败或产生错误结果。
```cpp
auto res = CommonPostProcess(groups, needChangeFormat, output, convOut, executor);
CHECK_RET(res == ACLNN_SUCCESS, res);
if(ConvTransposed2dSwitchHW){
    convOut = View4DSwapHWForTensor(convOut, executor); // swap后shape与output不匹配
}
auto result = l0op::ViewCopy(convOut, output, executor);
```
- **触发条件**: 在非 ASCEND910_95 平台上执行 transposed 2D 卷积，且满足 `isConvTransposed2dSwitchHW` 条件（stride[0]==1, pad==0, dilation==1, outW>4096, N==1, inC<=768, H==1）。
- **测试方案**: 在 910B 平台上构造满足条件的 ConvTranspose2d（如 input shape `[1, 768, 1, 64]`, weight `[768, 1, 1, 128]`, stride=[1,1], pad=[0,0], output_padding=[0,0]），验证输出正确性。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | L266-273 | 逻辑错误 | 高 | `All` 递归调用 `Any` 导致多参数校验逻辑退化为"首参满足+任一满足" |
| 2 | L1351 | 逻辑错误 | 高 | `<0 && ==0` 恒 false，weight 负数维度无法被拦截 |
| 3 | L67 | 语义错误 | 中 | `REFLECTION_MODE` 名称与值 `"constant"` 不匹配 |
| 4 | L130, L191 | 性能缺陷 | 中 | std::map 按值传递，每次调用产生不必要拷贝 |
| 5 | L2493-2496 | 代码质量 | 低 | outputW 初始赋值错误维度，依赖后续修正 |
| 6 | L3963-3967 | 逻辑错误 | 中 | SwapHW 在 PostProcess 后执行导致与 output shape 不匹配 |
