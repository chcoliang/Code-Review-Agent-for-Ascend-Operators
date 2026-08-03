# Ascend NPU 算子代码审查报告

**文件**: `aclnn_convolution.cpp`  
**审查日期**: 2026-08-03

---

### Bug 1: `All` 模板函数递归调用 `Any` 导致语义错误

- **位置**: 第 266-273 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `All` 函数本意是检查值是否满足所有参数列表的判断条件,但递归调用时错误地调用了 `Any` 而不是 `All`。这导致 `All` 实际语义变成"第一个满足且剩余任意一个满足",而非"所有都满足"。
- **触发条件**: 当使用 `CHECK_PARAM_ALL_GTE`、`CHECK_PARAM_LT_ALL`、`CHECK_PARAM_GT_ALL` 等宏且参数列表超过2个元素时,仅第一个和剩余中任一个被实际校验,其余参数跳过检查。例如 `CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, weightShapeN, weightShapeC)` 中只要 `inputShapeN>=0` 且 `inputShapeC/weightShapeN/weightShapeC` 任一个 `>=0` 即通过。
- **测试方案**: 构造一个 input tensor 其 N>=0, C>=0 但 weight N<0 (负值), 调用卷积接口,预期应报错 `ACLNN_ERR_PARAM_INVALID`,但实际可能放行。

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
```

---

### Bug 2: `CheckEmptyTensorTransposed` 中 weight shape 校验条件逻辑永假

- **位置**: 第 1351 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 条件 `if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))` 是一个永假表达式。`weightShape[i] < 0` 和 `weightShape[i] == 0` 不可能同时为真,因此整个 if 分支永远不会执行,weight 维度的负值和零值校验被完全跳过。
- **触发条件**: 在 ASCEND910_95 平台上,transposed=true 时,如果 weight 的空间维度为负值或为零(且对应 output 维度非零),本应报错但不会触发任何校验。
- **测试方案**: 在 ASCEND910_95 上构造 transposed conv,weight shape 某空间维度设为 -1 或 0 (output 对应维度非零),调用接口应报错但实际不报错。

```cpp
// 错误代码
if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))
// 修复: 应为 || 
if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0))
```

---

### Bug 3: 常量命名与实际值矛盾

- **位置**: 第 67 行
- **类型**: 语义错误/命名错误
- **严重程度**: 中
- **描述**: 常量名为 `REFLECTION_MODE` 但值为 `"constant"`。在第 2311 行作为 PadV3 的 mode 参数使用。如果调用者期望反射填充(reflection padding)但实际执行的是常量填充(constant padding),则计算结果错误;反之如果意图确实是常量填充,则命名具有严重误导性。
- **触发条件**: 当 C04 分支 weight 需要 pad 时(即 weight channel != 4),PadV3 使用该模式进行填充。如果后续开发者依赖变量名 `REFLECTION_MODE` 进行逻辑判断或修改,会引入错误。
- **测试方案**: 检查 PadV3 调用时 weight 的 padding 行为,确认是执行 constant padding(填充0)还是 reflection padding。对比两种 mode 的输出差异。

```cpp
// 问题代码
static const std::string REFLECTION_MODE = "constant";  // 名称与值矛盾
```

---

### Bug 4: `ConvL0Warper` 和 `L0FuncWarperByOpType` 按值传递 map 导致性能问题

- **位置**: 第 130 行, 第 192 行
- **类型**: 性能缺陷
- **严重程度**: 中
- **描述**: 两个函数均按值接收 `std::map<std::string, L0FUNCTION>` 参数,每次调用时会完整复制整个 map(包含所有已注册的 L0 函数指针)。由于 `FUNCTION_CALL` 和 `FUNCTION_CALL_BY_OPTYPE` 宏在卷积主路径上被频繁调用,这会带来不必要的内存分配和拷贝开销。
- **触发条件**: 每次卷积算子执行 Impl 阶段调用 L0 函数时都会触发。
- **测试方案**: 性能测试对比:将参数类型改为 `const std::map<std::string, L0FUNCTION>&` 后测量延迟变化。

```cpp
// 错误代码
static const aclTensor* ConvL0Warper(
    std::map<std::string, L0FUNCTION> l0Functions, ...)  // 按值传递
// 修复
static const aclTensor* ConvL0Warper(
    const std::map<std::string, L0FUNCTION>& l0Functions, ...)  // 按引用传递
```

---

### Bug 5: `Conv3dTo2dImpl` 私有成员 `l0Functions` 遮蔽基类成员

- **位置**: 第 3692 行
- **类型**: 代码缺陷/变量遮蔽
- **严重程度**: 低
- **描述**: `Conv3dTo2dImpl` 类在 private 区域重新声明了 `std::map<std::string, L0FUNCTION> l0Functions;`,与基类 `ConvolutionImpl` 的 protected 成员 `l0Functions` (第 3236 行) 同名。虽然当前代码因 PreProcess 和 Impl 都在同一派生类中使用了遮蔽后的版本所以功能正确,但这使得基类的 `l0Functions` 成为死代码,且任何未来的重构(如将某些逻辑提到基类)都会导致难以发现的 bug。
- **触发条件**: 当前不会导致运行时错误,但在代码维护和重构时容易引入隐蔽 bug。
- **测试方案**: 删除派生类的重复声明,使用基类的 protected 成员,验证功能不变。

---

### Bug 6: `isNotDMA` 函数中 `outputW` 初始赋值使用错误维度

- **位置**: 第 2493 行
- **类型**: 逻辑缺陷
- **严重程度**: 低
- **描述**: `outputW` 初始值为 `output->GetViewShape().GetDim(2)`,对于 NCHW 格式这是 H 维度而非 W。虽然紧接着的 if 判断在 4D 时会修正为 `GetDim(3)`,但如果 output 不是 4D(这在当前调用路径下不应发生),初始值就是错误的。代码防御性不足。
- **触发条件**: 当前调用路径下 output 始终为 4D NCHW,因此不会触发。但如果未来调用条件变化则会产生错误。
- **测试方案**: 在 CanSwitchC04 调用前构造非4D output(需绕过前置 check),观察 `isNotDMA` 行为异常。

---

### Bug 7: `DtypeCheckerTbc` 在 bias 为空时读取未初始化的 `bias.dataType`

- **位置**: 第 1026 行
- **类型**: 未初始化读取
- **严重程度**: 中
- **描述**: `DtypeCheckerTbc::Check` 中先无条件执行 `DataType biasDtype = engine.meta.bias.dataType;` (第 1026 行),然后才在第 1028 行判断 `engine.params.bias != nullptr`。当 bias 为 nullptr 时,`engine.meta.bias` 是默认构造的 `TensorMeta`,其 `dataType` 成员未被显式初始化(依赖默认值),读取行为虽不会崩溃但语义不正确。但实际上 `aclnnConvTbc` 接口要求 bias 非空(有空指针检查),所以当前路径下不会触发。
- **触发条件**: 如果 `CheckConvTbcParams` 被外部以 bias=nullptr 的方式调用(绕过空指针检查)。
- **测试方案**: 直接调用 `CheckConvTbcParams` 传入 bias=nullptr 的 engine,观察是否产生未定义行为。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | L266-273 | 逻辑错误 | 高 | `All` 递归调用 `Any`,多参数校验不完整 |
| 2 | L1351 | 逻辑错误 | 高 | `<0 && ==0` 永假,weight 维度校验失效 |
| 3 | L67 | 语义错误 | 中 | `REFLECTION_MODE = "constant"` 名值矛盾 |
| 4 | L130, L192 | 性能缺陷 | 中 | map 按值传递导致不必要拷贝 |
| 5 | L3692 | 变量遮蔽 | 低 | 派生类遮蔽基类同名成员 |
| 6 | L2493 | 逻辑缺陷 | 低 | outputW 初始值取错维度(H非W) |
| 7 | L1026 | 未初始化读取 | 中 | bias 为空时读取未初始化 dataType |
