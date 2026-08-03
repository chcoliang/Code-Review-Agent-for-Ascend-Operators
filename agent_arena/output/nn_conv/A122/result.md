# aclnn_convolution.cpp 代码审查报告

## Bug 列表

### Bug 1: `All` 模板函数递归调用错误 — 调用了 `Any` 而非 `All`

- **位置**: 第 266-273 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `All` 函数旨在检查"所有参数列表都满足判断条件"，但递归时调用了 `Any` 而非 `All`。这导致实际行为是：仅检查第一个元素满足条件 AND 剩余元素中任意一个满足条件，而非全部满足。`CHECK_PARAM_ALL_GTE` 和 `CHECK_PARAM_ALL_EQ` 等宏依赖此函数，导致参数校验逻辑不正确。
- **触发条件**: 当 `All` 的参数列表超过 2 个比较值时，中间值未被正确校验。例如 `CHECK_PARAM_ALL_GTE(0L, int64_t, a, b, c)` 中若 `a>=0, b<0, c>=0`，因第一个匹配后调用 `Any` 只需 b 或 c 任一满足即通过，b<0 不会被拦截。
- **测试方案**: 构造 input shape 为 `[1, -1, 3, 3]`（C为负数），weight shape 为 `[1, 1, 3, 3]`，验证 `CheckShape` 中的 `CHECK_PARAM_ALL_GTE(0L, int64_t, inputShapeN, inputShapeC, weightShapeN, weightShapeC)` 是否能正确拦截负值。

---

### Bug 2: `CheckEmptyTensorTransposed` 中逻辑条件永假

- **位置**: 第 1351 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 条件 `if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))` 永远为 false，因为 `weightShape[i] < 0` 和 `weightShape[i] == 0` 不可能同时成立。正确逻辑应为 `if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0))`，即 weight 维度为负时报错，或 weight 维度为 0 但 output 对应维度非 0 时报错。
- **触发条件**: 在 ASCEND910_95 平台上，transposed=true，weight 的空间维度为负值（如 `[4, 4, -1, 3, 3]`）时不会被拦截，可能导致后续计算异常。
- **测试方案**: 在 910_95 平台上构造 transposed conv，weight shape 含负维度，验证是否正常报错。

---

### Bug 3: `CheckParamsNullptrTbc` 空指针检查返回成功

- **位置**: 第 2121-2128 行
- **类型**: 错误处理缺陷
- **严重程度**: 高
- **描述**: `CHECK_RET(CheckNotNull(self, weight, bias, output), ACLNN_SUCCESS)` 当 `CheckNotNull` 返回 false 时，`CHECK_RET` 宏会返回第二个参数 `ACLNN_SUCCESS`。这意味着即使传入空指针，函数也返回成功，后续对空指针解引用会导致崩溃。应将 `ACLNN_SUCCESS` 改为 `ACLNN_ERR_PARAM_NULLPTR`。
- **触发条件**: 调用 `aclnnConvTbcGetWorkspaceSize` 时传入 `self=nullptr` 或 `weight=nullptr` 等。
- **测试方案**: 对 `aclnnConvTbcGetWorkspaceSize` 传入 nullptr 参数，验证是否返回错误码而非成功。

---

### Bug 4: `Conv3dTo2dImpl` 中 `l0Functions` 成员变量遮蔽基类成员

- **位置**: 第 3692 行
- **类型**: 变量遮蔽 (Shadowing)
- **严重程度**: 中
- **描述**: `Conv3dTo2dImpl` 在 private 中重新声明了 `std::map<std::string, L0FUNCTION> l0Functions;`，而基类 `ConvolutionImpl`（第 3236 行）已有同名 protected 成员。这导致 `PreProcess()` 中注册的 L0 函数写入了基类的 `l0Functions`，但如果有代码通过派生类的成员访问则会拿到空 map。当前流程中 `Impl()` 调用继承自基类的方法使用基类成员可能正常工作，但这是一个隐患。
- **触发条件**: 如果将来有代码在 `Conv3dTo2dImpl` 中直接访问私有 `l0Functions`，会得到空 map。
- **测试方案**: 在 310P 平台触发 `Conv3dTo2dImpl` 路径 (D==1, Kd==1, padD==0 的 3D conv)，确认 L0 函数调用正确。

---

### Bug 5: `PointWiseKernelBeyondLimits` 越界访问

- **位置**: 第 789-797 行
- **类型**: 数组越界
- **严重程度**: 中
- **描述**: 函数固定从 index 2 遍历到 `CONV_3D_DIM_SIZE`(5)，即访问 dim[2], dim[3], dim[4]。但该函数被 `Conv3dImpl::PreProcess` 调用时 input 可能是 4D tensor (CONV_2D_DIM_SIZE=4)，此时 dim[4] 越界。虽然当前调用者 `Conv3dImpl` 中 input 应为 5D，但函数本身缺乏防护。另外，在 `GetConvOpInfo` 的 `NeedPointWiseKernel` 调用处（line 2659），虽然由 conv2d 路径进入，但 `PointWiseKernelBeyondLimits` 对 4D fmap 会越界访问 `fmapShape.GetDim(4)`。
- **触发条件**: 在 conv2d splitW 分支中，`NeedPointWiseKernel` 返回 true 后调用 `PointWiseKernelBeyondLimits(input)`，此时 input 为 4D NCHW tensor。
- **测试方案**: 构造 4D input 满足 PointWise 条件 (stride=1, padding=0, dilation=1, groups=1, weight spatial=1)，验证是否触发越界。

---

### Bug 6: `REFLECTION_MODE` 常量命名与值不一致

- **位置**: 第 67 行
- **类型**: 语义错误/命名错误
- **严重程度**: 低
- **描述**: 常量命名为 `REFLECTION_MODE` 但值为 `"constant"`。在 PadV3 调用（第 2311 行）中使用了 `op::REFLECTION_MODE` 作为 padding mode 参数，实际传入的是 `"constant"` 模式。从上下文看此处需要的确实是 constant padding（填充 0 值），因此功能正确，但命名具有极强误导性。如果未来有人基于变量名使用它来做 reflection padding 将产生错误。
- **触发条件**: 维护者误用 `REFLECTION_MODE` 做 reflection padding。
- **测试方案**: 代码审查确认所有使用 `REFLECTION_MODE` 处是否确实需要 constant 模式。

---

### Bug 7: `CheckOutputBiasShape` 等函数返回值类型不匹配

- **位置**: 第 2130-2178 行
- **类型**: 类型不匹配
- **严重程度**: 低
- **描述**: `CheckOutputBiasShape`、`CheckOutputBiasDtype`、`CheckOutputBiasFormat` 的返回类型声明为 `aclnnStatus`，但错误分支返回 `false`（即 0），成功时返回 `true`（即 1）。在调用处（第 2182-2184 行）通过 `CHECK_RET(result, ACLNN_ERR_PARAM_INVALID)` 使用，恰好 `false=0` 视为失败触发返回错误码，`true=1` 视为成功。虽然在当前平台 `ACLNN_SUCCESS=0` 的假设下这可能不一致（true=1 不等于 ACLNN_SUCCESS=0），但因为 CHECK_RET 检查 bool 值不为 0 来判断成功，实际逻辑能正常运转。存在可移植性和可维护性风险。
- **触发条件**: 当 `ACLNN_SUCCESS` 的定义值发生变化时可能出现不兼容。
- **测试方案**: 单元测试 `aclnnConvTbcGetWorkspaceSize` 在 910_95 平台上传入不合法 output/bias shape 验证返回值。

---

### Bug 8: `ConvL0Warper` 按值传递 `std::map` 导致不必要的深拷贝

- **位置**: 第 130 行, 第 192 行
- **类型**: 性能缺陷
- **严重程度**: 低
- **描述**: `ConvL0Warper` 和 `L0FuncWarperByOpType` 的第一个参数 `std::map<std::string, L0FUNCTION> l0Functions` 按值传递，每次调用都会复制整个 map。应使用 `const std::map<std::string, L0FUNCTION>&` 传引用。
- **触发条件**: 每次卷积执行都会触发 map 拷贝。
- **测试方案**: 性能测试对比传引用前后的耗时。

---

## 汇总表

| # | 位置 | 类型 | 严重程度 | 简述 |
|---|------|------|----------|------|
| 1 | L266-273 | 逻辑错误 | 高 | `All` 递归调用 `Any`，多值校验失效 |
| 2 | L1351 | 逻辑错误 | 高 | `&&` 条件永假，weight负值校验失效 |
| 3 | L2121-2128 | 错误处理 | 高 | 空指针检查返回 ACLNN_SUCCESS |
| 4 | L3692 | 变量遮蔽 | 中 | 派生类重复声明基类同名成员 |
| 5 | L789-797 | 越界访问 | 中 | 4D tensor 固定访问 dim[4] |
| 6 | L67 | 命名错误 | 低 | REFLECTION_MODE 值为 "constant" |
| 7 | L2130-2178 | 类型不匹配 | 低 | aclnnStatus 函数返回 bool 值 |
| 8 | L130,192 | 性能缺陷 | 低 | std::map 按值传递 |
