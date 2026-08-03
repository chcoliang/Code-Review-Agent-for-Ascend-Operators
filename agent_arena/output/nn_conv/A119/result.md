# aclnn_convolution.cpp 代码审查报告

## Bug 列表

### Bug 1: `All` 模板函数递归调用逻辑错误

- **位置**: 第 266-273 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `All` 函数本应检查 value 满足与列表中**所有**元素的比较，但在第一次比较成功后，递归调用的是 `Any(value, f, list...)` 而非 `All(value, f, list...)`。这导致只要第一个元素满足条件，剩余元素中只需任一满足即通过，违反 "全部满足" 的语义。
- **触发条件**: 当使用 `CHECK_PARAM_ALL_GTE`、`CHECK_PARAM_LT_ALL`、`CHECK_PARAM_GT_ALL`、`CHECK_PARAM_ALL_EQ` 等宏，且参数列表包含3个或以上的比较值时，中间值校验被跳过。例如 `CHECK_PARAM_ALL_GTE(0L, int64_t, a, b, c)` 中若 a>=0 且 c>=0 但 b<0，检查仍会通过。
- **测试方案**: 构造 input shape 中 N>=0, C<0, weightN>=0 的 tensor，调用卷积接口，期望返回 `ACLNN_ERR_PARAM_INVALID`，实际会错误通过。

---

### Bug 2: `CheckEmptyTensorTransposed` 中不可能为真的条件表达式

- **位置**: 第 1351 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 条件 `if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))` 永远为假，因为一个值不可能同时小于 0 且等于 0。正确逻辑应为 `if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0))`，即 weight 维度小于 0 应报错，或者 weight 维度为 0 时 output 对应维度不为 0 也应报错。
- **触发条件**: 在 ASCEND910_95 平台上，transposed=true 时传入 weight 某空间维度为负数或为0(但output对应维度非0)的 tensor，校验不会拦截非法值。
- **测试方案**: 在 910_95 平台上构造 transposed conv，weight shape 中某维度为 -1 或为 0（output 对应维度为正），期望返回错误，实际不报错。

---

### Bug 3: 常量命名与值不匹配 (`REFLECTION_MODE` = "constant")

- **位置**: 第 67 行
- **类型**: 语义错误 / 命名错误
- **严重程度**: 中
- **描述**: 常量命名为 `REFLECTION_MODE`（反射填充），但实际值为 `"constant"`（常量填充）。在第 2310 行 `PadV3` 调用中使用该常量作为 mode 参数。如果意图是常量填充则功能正确但命名误导；如果意图是反射填充则产生错误结果。
- **触发条件**: C04 分支中 weight 的 C 维度不等于 4 时触发 PadV3 调用。
- **测试方案**: 检查 C04 分支下 PadV3 的填充行为是否符合预期（应为常量 0 填充）。如确认为常量填充，则仅需修正命名。

---

### Bug 4: `CheckNotNull` 函数未校验 `output` 参数

- **位置**: 第 2015-2022 行
- **类型**: 参数校验缺失
- **严重程度**: 中
- **描述**: 函数签名接收 `output` 参数但函数体内未对其做空指针检查。调用方 `CheckParamsNullptrTbc` (第 2124 行) 传入 output 并依赖此函数做校验，若 output 为 nullptr 后续将解引用崩溃。
- **触发条件**: 调用 `aclnnConvTbcGetWorkspaceSize` 时传入 output=nullptr。
- **测试方案**: 传入 output=nullptr 调用 aclnnConvTbc 接口，期望返回 ACLNN_ERR_PARAM_NULLPTR，实际会发生空指针解引用。

---

### Bug 5: `PointWiseKernelBeyondLimits` 对非5D tensor 越界访问

- **位置**: 第 789-797 行
- **类型**: 数组越界
- **严重程度**: 高
- **描述**: 函数循环从 `idx=CONST_VALUE_TWO(2)` 到 `idx < CONV_3D_DIM_SIZE(5)`，即访问 dim 索引 2、3、4。但该函数在 Conv3dImpl::PreProcess 中被调用时 `weight` 可以是 4D tensor（Conv2D 进入 3D PointWise 分支判断），访问 `fmapShape.GetDim(4)` 会越界。
- **触发条件**: 在非 IsSupportND() 且 IsSupportConv3DToConv3DV2() 的平台上，传入 4D 的 input tensor 且满足 PointWise 条件。
- **测试方案**: 在 910B 平台上构造 4D input (NCHW)、kernel 全为 1、stride=1、padding=0、dilation=1、groups=1 的卷积调用。

---

### Bug 6: `Conv3dTo2dImpl` 中 `l0Functions` 成员变量遮蔽

- **位置**: 第 3691 行
- **类型**: 变量遮蔽
- **严重程度**: 中
- **描述**: `Conv3dTo2dImpl` 类在 private 段声明了 `std::map<std::string, L0FUNCTION> l0Functions;`，遮蔽了基类 `ConvolutionImpl` 中已有的同名成员（第 3235 行）。在 `PreProcess` 中注册函数到基类的 `l0Functions`，但 `Impl` 中如果使用继承的通用逻辑则可能访问到空的局部 map 或反之。
- **触发条件**: 310P 平台上 Conv3D (D==1, Kd==1, padD==0) 走 Conv3dTo2d 分支时。
- **测试方案**: 在 310P 上构造符合 CanConv3dToConv2dOn310P 条件的 conv3d 调用，观察是否因找不到注册的 L0 函数而返回错误。

---

### Bug 7: 返回类型不匹配 (`aclnnStatus` 函数返回 bool)

- **位置**: 第 2129-2177 行 (`CheckOutputBiasShape`, `CheckOutputBiasDtype`, `CheckOutputBiasFormat`)
- **类型**: 类型不匹配
- **严重程度**: 中
- **描述**: 这三个函数声明返回 `aclnnStatus`（通常为 int 类型），但在实现中返回 `true`/`false`。虽然隐式转换为 0/1 可能碰巧与 ACLNN_SUCCESS 对应，但错误路径返回 `false`(0) 反而等于 ACLNN_SUCCESS，导致错误被静默忽略。
- **触发条件**: 当 output shape 维度检测失败、bias shape 不匹配、或 format 错误时，返回 false(=0=ACLNN_SUCCESS)，错误不被上层拦截。
- **测试方案**: 在 910_95 平台上调用 aclnnConvTbc，output 维度不为 3 或 bias format 非 ND，期望报错但实际可能通过。

---

### Bug 8: `ConvL0Warper` 和 `L0FuncWarperByOpType` 按值传递 map

- **位置**: 第 130 行, 第 192 行
- **类型**: 性能缺陷 / 潜在逻辑错误
- **严重程度**: 低
- **描述**: 两个函数将 `std::map<std::string, L0FUNCTION>` 按值传递，每次调用都会拷贝整个 map。应使用 `const std::map<std::string, L0FUNCTION>&` 传递引用。
- **触发条件**: 每次卷积执行时触发。
- **测试方案**: 性能测试对比传值与传引用的耗时差异。

---

### Bug 9: `ConstructPad` 对 conv1d 单元素 padding 的计算错误

- **位置**: 第 608 行
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: 当 `inputShape.size() == CONV_1D_DIM_SIZE` 且 `oldPad.size() == 1` 时，计算 `newPad = {oldPad[0] + oldPad[0]}`，即对称 padding 总和。但此值后续在 `InferShape` 中作为 `newPad[i]` 直接用于输出 shape 计算 `(input + newPad[i] - dilation*(weight-1) - 1) / stride + 1`。对于对称 padding，应为 `2 * pad`，这里结果相同；但对于 conv2d 的 2 元素 padding `{padH, padW}`，第 616 行 `newPad = {(oldPad[0] + oldPad[0]), (oldPad[1] + oldPad[1])}` 假设对称但实际可能不对称（4元素才表示不对称）。这与 PyTorch 的 conv2d padding 语义一致(2元素=对称)，所以这不是 bug。但当 `oldPad.size() == 2` 且 inputShape 是 1D 时，`newPad = {oldPad[0] + oldPad[1]}` 表示左右 padding 之和，这是正确的。**实际问题在 `CheckPad` 函数第 1462-1464 行**：对 1D/2D 使用 `newpad[i]` 作为总 padding 值，但 `inputShapeValueAfterPad` 的计算公式 `inputShapeValue + paddingValueFront - dilation*(weight-1) - 1` 把 newpad 当作总 padding(前+后)使用是正确的。此处无 bug，撤回。

---

### Bug 10: `aclnnConvDepthwise2dGetWorkspaceSize` 中 groups 初始化为1但后续可能未更新

- **位置**: 第 4500-4522 行
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: groups 初始值为 1，在 `CheckConvDepthwise2dParams` 中使用 groups=1 构建 ConvEngine 进行校验。但 depthwise conv 的 groups 应等于 input channels，groups=1 时 `ValueCheckerDepthwise2d` 中的 channel 校验可能产生误判。实际 groups 在第 4517 行重新赋值，但此时校验已完成。
- **触发条件**: depthwise conv2d 的 weight.C() 不为 1 且 inChannel 不等于 outChannel 时，由于 groups=1 校验逻辑可能与实际执行逻辑不一致。
- **测试方案**: 构造 depthwise conv 场景，input channels=64, weight shape=[64,1,3,3]，观察校验是否正确通过。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | L266-273 | 逻辑错误 | 高 | `All` 递归调用 `Any` 导致校验失效 |
| 2 | L1351 | 逻辑错误 | 高 | `&&` 导致条件永假，非法 weight 维度无法拦截 |
| 3 | L67 | 语义错误 | 中 | REFLECTION_MODE 值为 "constant" |
| 4 | L2015-2022 | 校验缺失 | 中 | CheckNotNull 未检查 output 空指针 |
| 5 | L789-797 | 数组越界 | 高 | 4D tensor 访问 dim(4) 越界 |
| 6 | L3691 | 变量遮蔽 | 中 | 子类 l0Functions 遮蔽基类成员 |
| 7 | L2129-2177 | 类型不匹配 | 中 | aclnnStatus 函数返回 bool，错误路径返回0=SUCCESS |
| 8 | L130,192 | 性能缺陷 | 低 | map 按值传递导致不必要拷贝 |
| 9 | L4500-4522 | 逻辑错误 | 中 | depthwise groups=1 做校验与实际执行不一致 |
