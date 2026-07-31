# aclnn_convolution.cpp 代码审查报告

## Bug 1: `All` 函数逻辑错误 — 递归调用了 `Any` 而非 `All`

- **位置**: 第 266-273 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `All` 模板函数用于检查所有参数是否满足条件，但在递归时错误地调用了 `Any` 而不是 `All`。这意味着对于3个及以上参数，`All` 实际语义变成了"第一个满足条件 AND 剩余任一满足条件"，而非"所有都满足条件"。
- **触发条件**: 当使用 `CHECK_PARAM_ALL_GTE` 宏（如第 1581 行检查 shape >= 0）时，若有3个以上参数且只有部分满足条件，应报错但不会报错。例如 `inputShapeN=1, inputShapeC=-1, weightShapeN=-1, weightShapeC=1`，由于 `Any` 找到 weightShapeC>=0 即返回 true，整个检查通过。
- **测试方案**: 构造 input shape = [1, -1, 5, 5]，weight shape = [-1, -1, 3, 3]，验证 `CheckShape` 是否正确拦截负值 shape。

## Bug 2: `CheckEmptyTensorTransposed` 中条件逻辑恒为 false

- **位置**: 第 1351 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 条件 `if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))` 永远无法为 true，因为 `weightShape[i] < 0` 和 `weightShape[i] == 0` 不可能同时成立。正确逻辑应为 `if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0))`（使用 `||` 而非 `&&`）。
- **触发条件**: 当 transposed=true 且 weight 的某个空间维度为负值时，本应报错的非法 weight shape 会通过校验，导致后续计算产生未定义行为。
- **测试方案**: 在 ASCEND910_95 上，构造 transposed conv 场景，weight shape 的空间维度为负值（如 [64, 32, -1, 3]），验证是否能正确拦截。

## Bug 3: `REFLECTION_MODE` 常量命名与值矛盾

- **位置**: 第 67 行
- **类型**: 语义错误/命名错误
- **严重程度**: 中
- **描述**: 常量命名为 `REFLECTION_MODE`（反射模式），但实际值为 `"constant"`（常量填充模式）。该常量在第 2311 行被传递给 `PadV3` 作为 padding mode 参数。如果意图是 constant 填充（用 0 填充），则命名错误会严重误导开发者；如果意图是 reflect 填充，则值错误会导致功能异常。
- **触发条件**: C04 分支中 weight 的 C 维度需要 pad 到 4 时（`CanSwitchC04` 返回 true 且 weight C != 4），PadV3 使用了错误的 mode。
- **测试方案**: 构造 input [1, 3, 224, 224]、weight [64, 3, 3, 3] 进入 C04 分支，检查 weight padding 行为是否符合预期。

## Bug 4: `PointWiseKernelBeyondLimits` 越界访问

- **位置**: 第 791-797 行
- **类型**: 数组越界
- **严重程度**: 高
- **描述**: 函数中循环上界固定为 `CONV_3D_DIM_SIZE`（5），即访问 dim index 2, 3, 4。但该函数被 `Conv3dImpl::PreProcess`（第 3728 行）调用时 input 可以是 4D tensor（如 NCHW），此时 `GetDim(4)` 会越界访问。实际调用点 Conv3d 场景下 input 确实是 5D，但函数本身缺乏维度检查，且 `NeedPointWiseKernel` 中 weight 也可能是 4D。
- **触发条件**: 虽然当前代码路径中 Conv3d 的 input 为 5D，但如果 `PointWiseKernelBeyondLimits` 被意外传入 4D tensor（如代码变更后），会导致未定义行为。
- **测试方案**: 验证 Conv3d PointWise 路径中 fmap 维度，确认 `GetDim(4)` 不会越界。

## Bug 5: Bias dtype 检查未使用 SoC 版本区分的支持列表

- **位置**: 第 998 行
- **类型**: 参数校验缺陷
- **严重程度**: 中
- **描述**: `DtypeChecker::Check` 中对 bias 的 dtype 检查使用了全局 `op::BIAS_SUPPORT_LIST`（包含 FP32/FP16/BF16），而非 `GetBiasDtypeSupportListBySocVersion()`。在 ASCEND310P 上 BF16 不被支持，但此处会错误地允许 BF16 bias 通过校验。
- **触发条件**: 在 ASCEND310P 平台上，传入 BF16 类型的 bias tensor，校验通过但实际硬件不支持，导致运行时错误。
- **测试方案**: 在 310P 平台构造 bias dtype 为 BF16 的卷积，验证是否能在 check 阶段拦截。

## Bug 6: `ConvL0Warper` 和 `L0FuncWarperByOpType` 以值传递 map

- **位置**: 第 130 行、第 192 行
- **类型**: 性能问题
- **严重程度**: 低
- **描述**: `ConvL0Warper` 和 `L0FuncWarperByOpType` 的第一个参数 `std::map<std::string, L0FUNCTION> l0Functions` 按值传递，每次调用都会完整拷贝整个 map。应使用 `const std::map<std::string, L0FUNCTION>&`。
- **触发条件**: 每次卷积操作的 Impl 阶段调用时都会触发不必要的 map 拷贝。
- **测试方案**: 性能测试，对比修改前后的 host 侧耗时。

## Bug 7: `aclnnConvDepthwise2dGetWorkspaceSize` 中 groups 参数不一致

- **位置**: 第 4501-4518 行
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: `groups` 初始化为 1 并传入 `ConvParams` 用于 check。`CheckConvDepthwise2dParams` 中的 `ValueCheckerDepthwise2d` 对 channel/groups 关系的校验使用了 groups=1，但实际 depthwise conv 的 groups 应为 input channel 数。check 通过后（第 4518 行）才将 groups 重新赋值。虽然 depthwise 的 weight C=1 使得 `weight.C() * groups == inChannel`（1*1 != inChannel 如果 inChannel != 1），但由于 ValueCheckerDepthwise2d 不检查 groups 关系（它检查的是 weight.C() == 0 or 1 和 outChannel % inChannel），该问题在当前 checker 组合下不会导致误判。然而语义上 groups=1 传入 ConvEngine 是不正确的。
- **触发条件**: 若后续 checker 增加对 groups 的校验逻辑，会因为 groups=1 而产生错误结果。
- **测试方案**: 构造 depthwise conv，groups != 1 场景，检查完整流程正确性。

## Bug 8: `isNotDMA` 函数中 `outputW` 初始赋值可能不正确

- **位置**: 第 2493-2496 行
- **类型**: 逻辑错误
- **严重程度**: 低
- **描述**: 第 2493 行 `outputW` 被赋值为 `output->GetViewShape().GetDim(2)`，在 NCHW 格式中这是 H 维度而非 W。紧接着第 2494-2496 行判断 `outputSize == CONV_2D_DIM_SIZE` 才修正为 `GetDim(3)`。但该函数仅在 `CanSwitchC04` 中被调用，调用前已确认 input 为 NCHW 4D，output 也应为 4D，因此 `outputSize` 总是等于 4，修正总会执行。不过代码中 `outputSize` 使用了 `GetDimNum()` 而非维度本身的值，逻辑上存在混淆风险。
- **触发条件**: 当前代码路径下不会触发问题（因为 C04 只支持 4D），但如果函数被复用到其他场景则会出错。
- **测试方案**: 确认 `isNotDMA` 只在 4D 场景下调用。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 266-273 | 逻辑错误 | 高 | `All` 函数错误调用 `Any`，导致"全部满足"变成"任一满足" |
| 2 | 1351 | 逻辑错误 | 高 | `&&` 连接的条件恒为 false，weight 负值 shape 无法被拦截 |
| 3 | 67 | 语义错误 | 中 | `REFLECTION_MODE` 命名但值为 `"constant"`，功能或命名有误 |
| 4 | 791-797 | 数组越界 | 高 | 循环固定到 dim 5，4D tensor 会越界 |
| 5 | 998 | 参数校验 | 中 | Bias dtype 检查未区分 SoC 版本，310P 上 BF16 漏检 |
| 6 | 130, 192 | 性能问题 | 低 | map 按值传递导致每次调用都完整拷贝 |
| 7 | 4501-4518 | 逻辑错误 | 中 | depthwise conv 以 groups=1 做 check，语义不正确 |
| 8 | 2493-2496 | 逻辑错误 | 低 | outputW 初始值取自 H 维度，依赖后续修正 |
