# Code Review: aclnn_scaled_masked_softmax.cpp (A146)

## Bug 1: 缺少对 x 的维度校验

- **位置**: 第 66-71 行 (`CheckTensorDim` 函数)
- **类型**: 逻辑遗漏
- **严重程度**: 高
- **描述**: `CheckTensorDim` 函数对 `mask` 和 `y` 进行了 4 维校验 (`INPUT_DIM_NUM`)，但遗漏了对输入 `x` 的维度校验。后续 `CheckShape` 函数中直接使用 `x->GetViewShape().GetDim(DIM_0)` 等访问第 0~3 维，如果 `x` 的维度不是 4，将导致越界访问或未定义行为。
- **触发条件**: 传入维度不为 4 的 `x` 张量（如 2D 或 5D 张量）。
- **测试方案**: 构造一个 3D 张量作为 `x`，4D 的 `mask` 和 `y`，调用 `aclnnScaledMaskedSoftmaxGetWorkspaceSize`，期望返回 `ACLNN_ERR_PARAM_INVALID` 而非崩溃。

## Bug 2: 错误日志中 dim 范围描述与实际逻辑不一致

- **位置**: 第 105 行
- **类型**: 日志/信息误导
- **严重程度**: 低
- **描述**: 错误日志写死为 `"Expected x and mask dim4 in range of (0, 4096]."`，但当平台为 `ASCEND910_95` 时，实际限制为 8192（`D_LIMIT_D`）。日志信息与真实限制不符，会误导用户排查问题。
- **触发条件**: 在 Ascend 910_95 平台上，当 `DIM_3` 大于 8192 时触发错误日志，但日志仍显示上限为 4096。
- **测试方案**: 在 910_95 平台上，传入 `DIM_3 = 5000` 的合法张量，验证不应报错；传入 `DIM_3 = 9000`，验证报错日志应显示正确上限 8192。

## Bug 3: fixedTriuMask 参数被硬编码忽略

- **位置**: 第 131-135 行
- **类型**: 功能限制/接口不一致
- **严重程度**: 中
- **描述**: 当 `fixedTriuMask` 为 `true` 时直接返回错误，而在调用内部函数时硬编码传入 `false`（第 135 行）。这意味着接口声明支持 `fixedTriuMask` 参数，但实现上完全拒绝了 `true` 的情况，且即使用户传入 `false`，内部也是硬编码 `false` 而非透传用户值。虽然当前逻辑上等价，但如果未来移除前面的检查，硬编码将导致参数丢失。
- **触发条件**: 用户传入 `fixedTriuMask = true`。
- **测试方案**: 调用时传入 `fixedTriuMask = true`，确认返回 `ACLNN_ERR_PARAM_INVALID` 及正确的错误信息。

## Bug 4: CheckShape 中未校验 x 与 y 的 shape 一致性

- **位置**: 第 74-110 行 (`CheckShape` 函数) 及第 112-121 行 (`CheckParams` 函数)
- **类型**: 逻辑遗漏
- **严重程度**: 中
- **描述**: `CheckShape` 仅校验了 `x` 和 `mask` 之间的形状关系，但完全没有校验输出 `y` 的 shape 是否与 `x` 一致。作为 softmax 算子，输出形状应与输入相同。如果 `y` 的 shape 与 `x` 不匹配，可能导致内存越界写入。
- **触发条件**: 传入 shape 与 `x` 不同的 `y` 张量（如 `x` 为 [2,4,8,16] 而 `y` 为 [1,1,1,1]）。
- **测试方案**: 构造 `x` shape 为 [2,4,8,16]，`y` shape 为 [2,4,8,8]，调用接口，期望返回形状不匹配错误。

## Bug 5: namespace 内声明 extern 函数包含在 extern "C" 块中

- **位置**: 第 25-27 行 (`extern "C"` 开始) 与第 29-123 行 (匿名 namespace)
- **类型**: 编译/链接问题
- **严重程度**: 中
- **描述**: `extern "C"` 块在第 26 行打开，匿名 namespace 在其内部（第 29 行）。在 `extern "C"` 内使用 C++ namespace、模板类（`std::initializer_list`）以及 C++ 特性是不规范的。更重要的是，第 39-44 行在匿名 namespace 内声明了 `extern` 函数，匿名 namespace 会给符号加上内部链接(internal linkage)，与 `extern` 语义矛盾，可能导致链接失败。
- **触发条件**: 链接时找不到 `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 和 `aclnnInnerScaledMaskedSoftmaxV2` 的符号定义。
- **测试方案**: 独立编译并链接此文件，检查是否出现 undefined reference 链接错误。

---

# 汇总表

| 编号 | 位置(行号) | 类型 | 严重程度 | 简要描述 |
|------|-----------|------|---------|---------|
| 1 | 66-71 | 逻辑遗漏 | 高 | 缺少对 x 的 4 维校验，可能导致越界访问 |
| 2 | 105 | 日志误导 | 低 | 错误日志中范围写死 4096，与 910_95 平台实际限制 8192 不符 |
| 3 | 131-135 | 功能限制 | 中 | fixedTriuMask 硬编码 false，接口语义不透明 |
| 4 | 74-121 | 逻辑遗漏 | 中 | 未校验 y 与 x 的 shape 一致性，可能导致内存越界 |
| 5 | 25-44 | 编译/链接 | 中 | 匿名 namespace 内 extern 声明与 extern "C" 块冲突，可能链接失败 |
