# aclnn_convolution.cpp 代码审查报告

## Bug 1: `All` 函数递归调用错误，应调用 `All` 却调用了 `Any`

- **位置**: 第 268-272 行
- **类型**: 逻辑错误
- **严重程度**: 高

**描述**:
`All` 模板函数本意是检查所有参数是否都满足条件，但递归时调用了 `Any` 而非 `All`。这导致实际语义变为"第一个元素满足条件 AND 剩余元素中任意一个满足条件"，而非"所有元素都满足条件"。

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

**触发条件**: 当使用 `CHECK_PARAM_ALL_GTE` 或 `CHECK_PARAM_ALL_EQ` 宏且参数列表超过2个元素时，如果第一个和任意一个后续元素满足条件但其他元素不满足，校验会误判为通过。例如 `CHECK_PARAM_ALL_GTE(0L, int64_t, a, b, c)` 当 a>=0, c>=0 但 b<0 时不会报错。

**测试方案**: 构造 input shape 为 `[1, -1, 3, 3]`（C=-1），weight shape 为 `[1, 1, 3, 3]`（N、C均>=0），调用 `CheckShape`。由于 All 函数错误，`CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, weightShapeN, weightShapeC)` 会漏检 inputShapeC < 0 的情况。

---

## Bug 2: `CheckEmptyTensorTransposed` 中条件逻辑恒为 false

- **位置**: 第 1351 行
- **类型**: 逻辑错误
- **严重程度**: 高

**描述**:
```cpp
if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))
```
`weightShape[i] < 0` 和 `weightShape[i] == 0` 不可能同时为真，因此整个条件永远为 false。正确逻辑应为 `||`：
```cpp
if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0))
```

**触发条件**: 在 transposed 模式下，ASCEND910_95 平台，weight 空间维度为负值或者 weight 维度为 0 但 output 对应维度非 0 时，应报错但不会报错，导致非法参数传入后续计算引发未定义行为。

**测试方案**: 在 ASCEND910_95 平台上，transposed=true，构造 weight shape 为 `[4, 4, -1, 3]` 或 `[4, 0, 3, 3]` 配合 output `[1, 4, 5, 5]`（对应维度非0），验证是否能正确拦截。

---

## Bug 3: `CommonPreProcess` 中 input 未执行 Contiguous 操作

- **位置**: 第 2217-2219 行
- **类型**: 逻辑错误
- **严重程度**: 高

**描述**:
当 `contiguous` 参数为 `true` 时，weight 和 bias 都调用了 `l0op::Contiguous`，但 input 仅被赋值为自身（`contiguousInput = input`），未调用 `l0op::Contiguous`。

```cpp
if (contiguous) {
    contiguousInput = input;  // BUG: 缺少 l0op::Contiguous(input, executor) 调用
    CHECK_RET(contiguousInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    ...
}
```

**触发条件**: 当输入 tensor 在内存中非连续（例如经过 slice/transpose 产生的 view tensor）时，后续 Conv L0 算子可能读取到错误数据或产生 segfault。

**测试方案**: 构造一个经 transpose 得到的非连续 input tensor（如先创建 NHWC tensor 再 permute 为 NCHW 但不做 contiguous），执行 conv2d，对比结果与连续 tensor 的结果。

---

## Bug 4: 变量名 `REFLECTION_MODE` 实际值为 `"constant"`

- **位置**: 第 67 行
- **类型**: 命名/语义错误
- **严重程度**: 中

**描述**:
```cpp
static const std::string REFLECTION_MODE = "constant";
```
变量命名为 `REFLECTION_MODE`（反射模式），但实际值为 `"constant"`（常量填充模式）。该变量在第 2311 行 `PadV3` 调用中用作 padding mode 参数。虽然当前场景（对 weight 做零填充）使用 `"constant"` 是正确的行为，但命名极易造成维护时的误用：若开发者依据变量名认为这是反射填充模式而复用，会导致功能错误。

**触发条件**: 后续维护者基于变量名误用该常量，或修改值为 `"reflect"` 以"修复"命名一致性时会破坏 C04 分支的 weight padding 逻辑。

**测试方案**: 代码审查 + 将变量名修改为 `CONSTANT_MODE` 或 `PAD_CONSTANT_MODE`，验证功能不变。

---

## Bug 5: `ConvL0Warper` 参数按值传递 map 造成性能浪费

- **位置**: 第 130 行
- **类型**: 性能问题
- **严重程度**: 低

**描述**:
```cpp
static const aclTensor* ConvL0Warper(
    std::map<std::string, L0FUNCTION> l0Functions, ...)  // 按值传递，每次调用拷贝整个 map
```
`l0Functions` 应使用 `const std::map<std::string, L0FUNCTION>&` 引用传递。同样的问题也出现在 `L0FuncWarperByOpType`（第 192 行）。每次调用卷积都会复制整个函数注册表 map。

**触发条件**: 每次卷积计算调用时都会触发，造成不必要的内存分配和拷贝。

**测试方案**: 性能 profiling，对比修改前后的 GetWorkspaceSize 耗时。

---

## Bug 6: `PointWiseKernelBeyondLimits` 越界访问风险

- **位置**: 第 789-797 行
- **类型**: 边界条件错误
- **严重程度**: 中

**描述**:
```cpp
static bool PointWiseKernelBeyondLimits(const aclTensor* fmap)
{
    auto fmapShape = fmap->GetViewShape();
    uint64_t dihiwi = 1;
    for (size_t idx = CONST_VALUE_TWO; idx < CONV_3D_DIM_SIZE; ++idx) {
        dihiwi = dihiwi * fmapShape.GetDim(idx);
    }
    return dihiwi >= MAX_UINT16;
}
```
该函数固定遍历到 `CONV_3D_DIM_SIZE`(=5)，但调用它时输入可能是 4D tensor（conv2d 场景下由 `Conv3dImpl::PreProcess` 中 `NeedPointWiseKernel` 后调用）。如果 fmap 是 4D tensor，`fmapShape.GetDim(4)` 越界访问。

**触发条件**: 在 conv3d 场景中，当 `NeedPointWiseKernel` 返回 true 后调用此函数。由于 `NeedPointWiseKernel` 从 idx=2 开始检查 weight，理论上 conv3d 才会进入，但函数本身没有维度校验。

**测试方案**: 用 4D input 手动调用 `PointWiseKernelBeyondLimits`，验证是否越界。

---

## Bug 7: `CheckOutputBiasShape` 返回类型与实际返回值不匹配

- **位置**: 第 2130-2152 行
- **类型**: 类型错误
- **严重程度**: 低

**描述**:
函数声明返回 `aclnnStatus`，但函数体中 `return false` 和隐含的 `return true`（第 2151 行）返回的是布尔值。依赖隐式类型转换 `false->0`（即 ACLNN_SUCCESS）和 `true->1`。虽然可能碰巧工作，但这种依赖未定义的枚举值映射是危险的。

**触发条件**: 如果 `ACLNN_SUCCESS != 0` 或 `ACLNN_ERR_PARAM_INVALID != 1`，逻辑就会出错。

**测试方案**: 静态分析工具检测，或确认 aclnnStatus 枚举定义。

---

## Bug 8: `isNotDMA` 函数对非 4D tensor 的越界访问

- **位置**: 第 2488-2493 行
- **类型**: 边界条件错误
- **严重程度**: 中

**描述**:
```cpp
int64_t inputHeight = (int64_t)input->GetViewShape().GetDim(2);
int64_t inputWidth = (int64_t)input->GetViewShape().GetDim(3);
int64_t weightH = (int64_t)weight->GetViewShape().GetDim(2);
int64_t weightW = (int64_t)weight->GetViewShape().GetDim(3);
```
函数直接访问 dim(2) 和 dim(3)，假设输入是 4D。虽然调用链中 `CanSwitchC04` 会先判断 format 为 NCHW，但函数本身没有维度保护。

此外第 2493 行：
```cpp
int64_t outputW = (int64_t)output->GetViewShape().GetDim(2);  // 取的是 H 而非 W
if (outputSize == CONV_2D_DIM_SIZE) {
    outputW = static_cast<int64_t>(output->GetViewShape().GetDim(3));
}
```
初始赋值取 Dim(2) 是 H 维度（对 NCHW 格式），当 output 恰好为 4D 时会被覆盖。但如果 `outputSize != 4`（不应发生但缺乏保护），`outputW` 将是错误的 H 值。

**触发条件**: 当 output dim != 4 时（理论上不应发生于 C04 路径，但缺乏防御性检查）。

**测试方案**: 在 CanSwitchC04 的调用前 mock 一个非 4D output，验证行为。

---

# 汇总表

| # | 行号 | 类型 | 严重程度 | 简述 |
|---|------|------|----------|------|
| 1 | 268-272 | 逻辑错误 | 高 | `All` 递归调用 `Any` 导致参数校验不完整 |
| 2 | 1351 | 逻辑错误 | 高 | `<0 && ==0` 恒为 false，weight 负值无法检出 |
| 3 | 2217-2219 | 逻辑错误 | 高 | input 未执行 Contiguous，非连续 tensor 计算错误 |
| 4 | 67 | 命名错误 | 中 | `REFLECTION_MODE` 值为 "constant"，易误用 |
| 5 | 130, 192 | 性能问题 | 低 | map 按值传递，每次调用产生拷贝 |
| 6 | 789-797 | 边界条件 | 中 | 固定遍历到 5D 索引，4D tensor 越界风险 |
| 7 | 2130-2152 | 类型错误 | 低 | 返回 bool 但声明返回 aclnnStatus |
| 8 | 2488-2493 | 边界条件 | 中 | 硬编码 dim 索引缺乏维度保护 |
