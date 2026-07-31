# aclnn_convolution.cpp 代码审查报告

## Bug 1: `All` 函数递归调用错误导致校验逻辑不完整

- **位置**: 第 266-273 行
- **类型**: 逻辑错误
- **严重程度**: 高

**描述**:
`All` 模板函数的目的是检查所有参数是否满足给定条件，但递归时错误地调用了 `Any` 而非 `All`。这导致：仅第一个元素被严格校验，剩余元素只要有任意一个满足条件就通过。

```cpp
template <typename T, typename Func, typename... LIST>
static inline bool All(T value, Func f, T compare, LIST... list)
{
    bool result = f(value, compare);
    if (result) {
        return Any(value, f, list...); // BUG: 应为 All(value, f, list...)
    }
    return false;
}
```

**触发条件**:
当使用 `CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, weightShapeN, weightShapeC)` 时，实际校验逻辑为 `(inputShapeN >= 0) && (inputShapeC >= 0 || weightShapeN >= 0 || weightShapeC >= 0)`，而非期望的全部 >= 0。如果 inputShapeN >= 0 且仅有一个其他值 >= 0（其余为负），校验会错误通过。

**测试方案**:
构造输入使 inputShapeN=1, inputShapeC=-1, weightShapeN=-1, weightShapeC=1，观察是否能绕过形状校验进入计算流程导致异常。

---

## Bug 2: `CheckEmptyTensorTransposed` 中条件逻辑永假

- **位置**: 第 1351 行
- **类型**: 逻辑错误
- **严重程度**: 高

**描述**:
条件 `if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))` 永远为 false，因为 `weightShape[i] < 0` 与 `weightShape[i] == 0` 互斥。正确逻辑应使用 `||`。

```cpp
// 错误代码：
if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))

// 正确代码应为：
if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0))
```

**触发条件**:
在 transposed 模式下，当 weight 某个空间维度为负值或为 0 但 output 对应维度非 0 时，校验无法拦截非法输入，可能导致后续计算行为未定义。

**测试方案**:
设置 transposed=true，weight shape 中某一空间维度设为 -1 或设为 0 而 output 对应维度不为 0，验证是否能被正确拦截。

---

## Bug 3: 常量命名与值不一致 (REFLECTION_MODE)

- **位置**: 第 67 行
- **类型**: 语义错误/命名错误
- **严重程度**: 中

**描述**:
常量名为 `REFLECTION_MODE` 暗示反射填充模式，但其值为 `"constant"`（常量填充）。该常量在第 2310 行被用于 `PadV3` 调用。

```cpp
static const std::string REFLECTION_MODE = "constant"; // 名称与值矛盾
```

**触发条件**:
当 C04 分支中 weight 需要 pad 到 4 通道时，使用 `REFLECTION_MODE` 作为 PadV3 的 mode 参数。如果开发者本意是反射填充，则 padding 结果将错误。

**测试方案**:
在 C04 场景下，输入 cin=1 的 weight，检查 pad 后的 weight 值是否为 0 填充（constant）还是反射填充。对比两种填充模式的精度结果。

---

## Bug 4: `ConvL0Warper` 按值传递 map 导致性能问题

- **位置**: 第 130 行、第 192 行
- **类型**: 性能缺陷
- **严重程度**: 低

**描述**:
`ConvL0Warper` 和 `L0FuncWarperByOpType` 的第一个参数 `std::map<std::string, L0FUNCTION> l0Functions` 按值传递，每次调用都会完整拷贝整个 map。应使用 `const std::map<std::string, L0FUNCTION>&`。

**触发条件**:
每次调用卷积实现时都会触发 map 拷贝，在高频调用场景下造成不必要的内存分配和拷贝开销。

**测试方案**:
在性能测试中测量 ConvL0Warper 调用前后的耗时，对比改为引用传递后的性能。

---

## Bug 5: `CheckNotNull` 未校验 output 参数

- **位置**: 第 2016-2021 行
- **类型**: 参数校验缺失
- **严重程度**: 中

**描述**:
函数签名接收 output 参数但未对其进行空指针检查。如果 output 为 nullptr，后续代码会崩溃。

```cpp
static inline bool CheckNotNull(
    const aclTensor* self, const aclTensor* weight, const aclTensor* bias, const aclTensor* output)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(bias, return false);
    // 缺少: OP_CHECK_NULL(output, return false);
    return true;
}
```

**触发条件**:
在 `aclnnConvTbcGetWorkspaceSize` 中，如果 output 传入 nullptr，`CheckParamsNullptrTbc` 不会报错，后续访问 output 成员会导致段错误。

**测试方案**:
调用 `aclnnConvTbcGetWorkspaceSize` 时传入 output=nullptr，观察是否正确返回错误码而非崩溃。

---

## Bug 6: `aclnnConvDepthwise2dGetWorkspaceSize` 中 groups 初始值用于校验

- **位置**: 第 4500-4508 行
- **类型**: 逻辑错误
- **严重程度**: 中

**描述**:
`groups` 初始设为 1 并用此值构造 `ConvParams` 进行参数校验。但 depthwise 卷积的 groups 应等于输入通道数。虽然 `ValueCheckerDepthwise2d` 自身不直接使用 groups，但 `ConvXdChecker` 中调用 `CalcOutputShape()` 时使用了 `engine.params.groups`（即 1）来计算 transposed 输出形状，如果误走入该路径将导致 output shape 计算错误。

```cpp
int64_t groups = 1;  // 错误的初始值
ConvParams params = {..., groups, ...};
ConvEngine convEngine(params);
ret = CheckConvDepthwise2dParams(convEngine);  // 使用 groups=1 校验
```

**触发条件**:
当 depthwise 卷积通过 `ConvXdChecker` 校验推断 output shape 时（虽然 transposed=false），计算 `cOut = meta.weight.N()` 不受影响。但如果 `CheckChannelAndGroups` 被其他路径调用，`weight.C() * groups != inChannel` 校验使用 groups=1 可能误报。

**测试方案**:
构造 depthwise 卷积场景（inChannel=64, weight shape=[64,1,3,3]），验证校验和计算结果是否正确。

---

## Bug 7: `PointWiseKernelBeyondLimits` 越界访问

- **位置**: 第 789-797 行
- **类型**: 边界条件/越界访问
- **严重程度**: 中

**描述**:
函数中循环从 `CONST_VALUE_TWO`(=2) 到 `CONV_3D_DIM_SIZE`(=5)，通过 `fmapShape.GetDim(idx)` 访问维度。但如果输入 fmap 是 4D (NCHW)，`GetDim(4)` 会越界。

```cpp
for (size_t idx = CONST_VALUE_TWO; idx < CONV_3D_DIM_SIZE; ++idx) {
    dihiwi = dihiwi * fmapShape.GetDim(idx);  // 当 fmap 是 4D 时, idx=4 越界
}
```

**触发条件**:
当 `NeedPointWiseKernel` 返回 true 且输入为 4D tensor 时（Conv3dImpl 调用，但 input 被错误地以 4D 传入），会发生越界访问。

**测试方案**:
构造 4D 输入满足 pointwise 条件（groups=1, stride=1, padding=0, dilation=1, weight 空间维均为 1），验证是否崩溃。

---

## Bug 8: `isNotDMA` 中 stride/dilation 数组越界风险

- **位置**: 第 2503-2506 行
- **类型**: 边界条件
- **严重程度**: 低

**描述**:
直接通过 `(*stride)[1]` 和 `(*dilation)[1]` 访问第 2 个元素，但未检查数组是否至少有 2 个元素。如果 stride 或 dilation 长度为 1（如 conv1d 转为 conv2d 前），将越界。

**触发条件**:
理论上 `isNotDMA` 仅在 conv2d 路径调用（stride/dilation 已被扩展为 2 维），但如果被错误复用于 1D 场景则会触发。

**测试方案**:
直接用 size=1 的 stride/dilation 调用 `isNotDMA` 观察行为。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 266-273 | 逻辑错误 | 高 | `All` 递归误调 `Any`，多参数校验不完整 |
| 2 | 1351 | 逻辑错误 | 高 | `&&` 连接互斥条件，判断永假，校验失效 |
| 3 | 67 | 语义错误 | 中 | REFLECTION_MODE 值为 "constant"，名实不符 |
| 4 | 130, 192 | 性能缺陷 | 低 | map 按值传递，频繁不必要拷贝 |
| 5 | 2016-2021 | 校验缺失 | 中 | `CheckNotNull` 遗漏 output 空指针检查 |
| 6 | 4500-4508 | 逻辑错误 | 中 | depthwise groups=1 用于校验，值不正确 |
| 7 | 789-797 | 越界访问 | 中 | 4D tensor 访问 dim(4) 越界 |
| 8 | 2503-2506 | 边界条件 | 低 | stride/dilation 未检查 size 直接访问 [1] |
