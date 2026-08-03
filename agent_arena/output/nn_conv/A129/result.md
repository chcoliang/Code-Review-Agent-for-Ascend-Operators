# Ascend NPU 算子代码审查报告

## 文件: aclnn_convolution.cpp

---

### Bug 1: `All` 模板函数递归调用 `Any` 导致逻辑错误

- **位置**: 第 266-273 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `All` 函数本意是检查 `value` 对参数列表中的**所有**元素都满足比较条件 `f`，但递归时调用了 `Any` 而非 `All`。这导致实际只检查第一个元素满足条件后，剩余元素只要**任一**满足即返回 true，而非要求全部满足。
- **触发条件**: 当使用 `CHECK_PARAM_ALL_GTE`、`CHECK_PARAM_ALL_EQ`、`CHECK_PARAM_GT_ALL`、`CHECK_PARAM_LT_ALL` 宏，且参数列表有3个或更多元素时，第一个满足条件但后续存在不满足条件的元素时，校验会被错误跳过。例如 `CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, weightShapeN, weightShapeC)` 当 `inputShapeN>=0` 且 `inputShapeC>=0` 但 `weightShapeN<0` 时，只要 `weightShapeC>=0` 就不会报错。
- **测试方案**: 构造 input shape = [1, 2, 3, 3]，weight shape = [1, -1, 3, 3]（C 为负数但 N>=0），观察是否能通过 shape 校验（预期应报错但实际不会）。

```cpp
// 错误代码
template <typename T, typename Func, typename... LIST>
static inline bool All(T value, Func f, T compare, LIST... list)
{
    bool result = f(value, compare);
    if (result) {
        return Any(value, f, list...);  // BUG: 应为 All
    }
    return false;
}

// 修复
template <typename T, typename Func, typename... LIST>
static inline bool All(T value, Func f, T compare, LIST... list)
{
    bool result = f(value, compare);
    if (result) {
        return All(value, f, list...);
    }
    return false;
}
```

---

### Bug 2: `CheckEmptyTensorTransposed` 中 weight 维度校验条件恒为 false

- **位置**: 第 1350 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 条件 `if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))` 要求 `weightShape[i]` 同时小于 0 **且**等于 0，这在数学上不可能成立，因此该分支永远不会被执行。这意味着 transposed 模式下 weight 的非法维度值（负数或不匹配的零）不会被正确拦截。
- **触发条件**: 在 ASCEND910_95 平台上，transposed=true 时，weight 某个空间维度为负值或为 0（但 output 对应维度非 0），该非法输入会绕过校验。
- **测试方案**: 在 ASCEND910_95 平台构造 transposed conv2d，weight shape = [4, 2, -1, 3]（H 为负数），验证是否报错（预期应报错但不会）。

```cpp
// 错误代码
if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0)) {

// 修复: 应为 OR 逻辑
if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0)) {
```

---

### Bug 3: 常量命名与值不匹配 (`REFLECTION_MODE` = "constant")

- **位置**: 第 67 行
- **类型**: 命名/语义错误
- **严重程度**: 中
- **描述**: 常量命名为 `REFLECTION_MODE` 但实际值为 `"constant"`。在第 2310 行被用于 `PadV3` 的 mode 参数。该命名极易导致维护者误解语义，若后续有人将其用于真正需要 "reflect" 模式的场景会产生计算错误。
- **触发条件**: 当前不直接产生运行时错误（因为 PadV3 零填充逻辑需要 "constant" 模式），但任何基于变量名语义的使用都会出错。
- **测试方案**: 代码走读确认。在 C04 分支中 PadV3 调用处确认 mode="constant" 是正确意图，然后修复命名。

```cpp
// 错误代码
static const std::string REFLECTION_MODE = "constant";

// 修复: 修正变量名
static const std::string CONSTANT_PAD_MODE = "constant";
```

---

### Bug 4: `ConvL0Warper` 和 `L0FuncWarperByOpType` 按值传递 `std::map`

- **位置**: 第 130 行, 第 192 行
- **类型**: 性能缺陷
- **严重程度**: 中
- **描述**: 函数参数 `std::map<std::string, L0FUNCTION> l0Functions` 按值传递，导致每次调用时整个 map 被完整拷贝（包含所有注册的函数指针）。在卷积的热路径上，这会带来不必要的内存分配和拷贝开销。
- **触发条件**: 每次执行卷积操作调用 `FUNCTION_CALL` 或 `FUNCTION_CALL_BY_OPTYPE` 宏时均会触发。
- **测试方案**: 性能测试对比。修改为 const 引用后测量单次卷积调用的额外延迟减少量。

```cpp
// 错误代码
static const aclTensor* ConvL0Warper(
    std::map<std::string, L0FUNCTION> l0Functions, ...)

// 修复
static const aclTensor* ConvL0Warper(
    const std::map<std::string, L0FUNCTION>& l0Functions, ...)
```

---

### Bug 5: `CheckPadTbc` 缺少 2 倍 padding 因子

- **位置**: 第 1693 行
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: TBC 卷积中，`pad` 参数表示在输入两侧各填充 `pad` 个元素，因此有效输入长度应为 `inputL + 2*pad`。但 `CheckPadTbc` 中校验公式为 `inputShapeL + padding[0] - (weightShapeL - 1) - 1 >= 0`，缺少 2 倍因子。正确应为 `inputShapeL + 2*padding[0] - (weightShapeL - 1) - 1 >= 0`。这会导致某些本应合法的输入被错误拒绝。
- **触发条件**: 当 `inputL + pad < weightL` 但 `inputL + 2*pad >= weightL` 时（如 inputL=3, pad=2, weightL=6），合法输入会被错误拦截。
- **测试方案**: 构造 conv_tbc 输入 shape=[5,1,4]（T=5,B=1,C=4），weight shape=[4,4,3]（kernel=3），pad=1。预期有效长度=5+2=7>=3合法，但检查公式 5+1-2-1=3>=0 也通过。再用 inputL=2, weightL=4, pad=1: 有效长度=2+2=4>=4合法，但检查 2+1-3-1=-1<0 会错误拒绝。

```cpp
// 错误代码
int64_t inputShapeValueAfterPad = (inputShapeL + padding[0] - dilationValue * (weightShapeL - 1) - 1);

// 修复
int64_t inputShapeValueAfterPad = (inputShapeL + 2 * padding[0] - dilationValue * (weightShapeL - 1) - 1);
```

---

### Bug 6: `Conv3dTo2dImpl` 类中重复声明 `l0Functions` 成员变量

- **位置**: 第 3691 行
- **类型**: 遮蔽/冗余错误
- **严重程度**: 低
- **描述**: `Conv3dTo2dImpl` 类在 private 区域重新声明了 `std::map<std::string, L0FUNCTION> l0Functions`（第3691行），但基类 `ConvolutionImpl` 已有同名受保护成员（第3235行）。这会导致 `PreProcess` 中注册的函数写入本地成员，而 `Impl` 中（若使用基类方法/宏）可能访问基类的空 map，但因该类 `Impl` 直接用了 `CommonConvImpl` 并传入 `l0Functions`，实际访问的是局部的。此处为代码维护隐患。
- **触发条件**: 当前不直接导致运行时错误（因当前实现恰好使用了正确的成员），但如果后续重构或调用基类方法时可能引入问题。
- **测试方案**: 删除子类的重复声明，运行 Conv3dTo2d 的全量测试验证功能不变。

---

## 汇总表

| # | 位置 | 类型 | 严重程度 | 简述 |
|---|------|------|----------|------|
| 1 | 第 266-273 行 | 逻辑错误 | 高 | `All` 递归调用 `Any`，导致参数校验只检查部分条件 |
| 2 | 第 1350 行 | 逻辑错误 | 高 | `< 0 && == 0` 恒假，weight 维度非法值无法被拦截 |
| 3 | 第 67 行 | 命名错误 | 中 | `REFLECTION_MODE` 值为 "constant"，语义矛盾 |
| 4 | 第 130, 192 行 | 性能缺陷 | 中 | `std::map` 按值传递导致不必要的拷贝 |
| 5 | 第 1693 行 | 逻辑错误 | 中 | TBC padding 校验缺少 2 倍因子，误拒合法输入 |
| 6 | 第 3691 行 | 成员遮蔽 | 低 | 子类重复声明基类同名成员变量 |
