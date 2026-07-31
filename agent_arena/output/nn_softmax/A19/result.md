# Code Review: aclnn_softmax.cpp

## Bug 1: `out` 参数未做空指针校验
- **位置**: 行 32-37 (`CheckNotNull` 函数)
- **类型**: 参数校验缺失
- **严重程度**: 高
- **描述**: `CheckNotNull` 函数仅检查了 `self` 是否为空指针，对 `out` 参数使用 `(void)out;` 显式忽略。后续在行 62 `OP_CHECK_DTYPE_NOT_SUPPORT(out, ...)`、行 83 `OP_CHECK_SHAPE_NOT_EQUAL(self, out, ...)`、行 137 `out->GetDataType()`、行 141 `ViewCopy(cast_out, out, ...)` 均会解引用 `out`，当 `out` 为 nullptr 时导致段错误崩溃。
- **触发条件**: 调用 `aclnnSoftmaxGetWorkspaceSize(self, 0, nullptr, &workspaceSize, &executor)`，其中 `self` 为有效非空 tensor。
- **测试方案**: 构造有效 float32 shape=[2,3] 的 self tensor，out 传 nullptr，预期应返回 `ACLNN_ERR_PARAM_NULLPTR` 而非崩溃。

## Bug 2: 空 tensor 提前返回跳过 `out` 的全部校验
- **位置**: 行 92-95 (`CheckParams` 函数)
- **类型**: 逻辑错误/参数校验不完整
- **严重程度**: 中
- **描述**: 当 `self->IsEmpty()` 为 true 时，函数直接返回 `ACLNN_SUCCESS`，跳过了对 `out` 的 dtype 检查、dim 检查和 shape 一致性检查。如果 `out` 为 nullptr、dtype 不支持或 shape 不匹配，均不会被检测到，返回了错误的成功状态。
- **触发条件**: self 为空 tensor (shape=[0,3])，out 为 nullptr 或 dtype=int8 或 shape=[2,3]。
- **测试方案**: 构造 self shape=[0,3] dtype=float32，out 设为 nullptr 或 shape=[2,3] 的 int8 tensor，预期应返回错误码，实际返回 ACLNN_SUCCESS。

## Bug 3: `workspaceSize` 和 `executor` 输出指针未做空指针校验
- **位置**: 行 108-109 (`aclnnSoftmaxGetWorkspaceSize` 函数入口)
- **类型**: 参数校验缺失
- **严重程度**: 高
- **描述**: `workspaceSize` 和 `executor` 作为输出参数指针，在行 123/145 处 `*workspaceSize = 0` 以及行 124/146 处 `uniqueExecutor.ReleaseTo(executor)` 被解引用，但函数入口未对它们进行空指针检查，传入 nullptr 将导致段错误。
- **触发条件**: 调用 `aclnnSoftmaxGetWorkspaceSize(self, 0, out, nullptr, &executor)` 或 `aclnnSoftmaxGetWorkspaceSize(self, 0, out, &ws, nullptr)`。
- **测试方案**: 传入有效的 self/out/dim，将 workspaceSize 设为 nullptr，预期应返回错误码而非崩溃。

## Bug 4: 负数 `dim` 未归一化直接传递给底层算子
- **位置**: 行 133 (`l0op::SoftmaxV2(self_contiguous, dim, ...)`)
- **类型**: 边界条件/参数传递
- **严重程度**: 中
- **描述**: `CheckDim` 函数(行66-78)验证了 dim 在 `[-selfDimNum, selfDimNum)` 范围内合法，但未将负数 dim 归一化为非负值（即 `dim += selfDimNum`）。PyTorch 语义下 dim=-1 表示最后一维，如果底层 `SoftmaxV2` 算子不处理负数 axis，将产生未定义行为或运行时错误。
- **触发条件**: self shape=[2,3,4]，dim=-1（期望在 axis=2 上做 softmax）。
- **测试方案**: 构造 3 维 float32 tensor，dim=-1，对比 dim=2 的结果，验证是否一致或是否报错。

## Bug 5: 注释与实际调用不匹配
- **位置**: 行 132
- **类型**: 代码质量/注释错误
- **严重程度**: 低
- **描述**: 注释写"调用SoftmaxGrad算子kernel"，但实际调用的是 `l0op::SoftmaxV2`（前向 Softmax），并非梯度算子。误导代码维护者。
- **触发条件**: 代码审查即可发现。
- **测试方案**: 人工审查，确认注释与实际调用语义一致。

## Bug 6: 910B dtype 支持列表包含 DT_DOUBLE 但 AIC 可能不支持
- **位置**: 行 39-45
- **类型**: 配置与注释不一致/平台兼容性
- **严重程度**: 中
- **描述**: 行 40 注释明确说明 "AIC支持:DT_BF16, DT_FLOAT16, DT_FLOAT, AICPU支持 DT_DOUBLE"，即 DT_DOUBLE 仅 AICPU 支持。但 `ASCEND910B_DTYPE_SUPPORT_LIST` 包含 DT_DOUBLE 且代码中调用的是 `SoftmaxV2`（通常在 AI Core 上执行）。如果 SoftmaxV2 不支持 DT_DOUBLE，传入 double 类型将在运行时失败，dtype 校验无法拦截。
- **触发条件**: 在 Ascend 910B 平台上传入 DT_DOUBLE 类型 tensor。
- **测试方案**: 在 910B 平台构造 double 类型 shape=[2,3] 的 tensor，调用 softmax，验证是否能正确执行。

## 汇总
| # | 位置 | 类型 | 严重程度 | 描述 |
|---|------|------|----------|------|
| 1 | 行 32-37 | 参数校验缺失 | 高 | out 未做空指针检查，解引用导致段错误 |
| 2 | 行 92-95 | 逻辑错误 | 中 | 空 tensor 时跳过 out 全部校验 |
| 3 | 行 108-109 | 参数校验缺失 | 高 | workspaceSize/executor 指针未校验 |
| 4 | 行 133 | 边界条件 | 中 | 负数 dim 未归一化直接传给底层算子 |
| 5 | 行 132 | 注释错误 | 低 | 注释写 SoftmaxGrad 实际是 SoftmaxV2 |
| 6 | 行 39-45 | 平台兼容性 | 中 | DT_DOUBLE 在支持列表中但 AIC 可能不支持 |
