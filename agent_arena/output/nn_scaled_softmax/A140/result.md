# Code Review: aclnn_scaled_masked_softmax.cpp

## Bug 1: CheckNotNull 校验失败返回错误码不正确

- **位置**: 第 115 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `CHECK_RET(CheckNotNull(x, mask, y), ACLNN_SUCCESS)` 当 `CheckNotNull` 返回 `false`（即输入为空指针）时，宏返回 `ACLNN_SUCCESS`，这意味着空指针校验失败却返回成功状态码，调用者无法感知错误，后续使用空指针会导致崩溃。应返回 `ACLNN_ERR_PARAM_INVALID` 或类似的错误码。
- **触发条件**: 传入 `x`、`mask` 或 `y` 为 `nullptr`。
- **测试方案**: 调用 `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 时传入 nullptr 参数，验证返回值应为错误码而非 `ACLNN_SUCCESS`。

## Bug 2: 错误日志中 DIM_3 范围描述与实际限制不一致

- **位置**: 第 106 行
- **类型**: 日志/文档错误
- **严重程度**: 低
- **描述**: 错误日志固定输出 `"Expected x and mask dim4 in range of (0, 4096]."` ，但当平台为 ASCEND910_95 时，实际限制为 8192。日志信息与实际判断逻辑不一致，会误导调试人员。
- **触发条件**: 在 ASCEND910_95 平台上，当 `DIM_3` 值在 (4096, 8192] 范围内触发错误日志时（不会触发，但如果超过 8192 则触发），日志信息中写的是 4096 而实际应为 8192。
- **测试方案**: 在 ASCEND910_95 平台上传入 `DIM_3 > 8192` 的 tensor，检查错误日志输出是否正确反映实际限制值 8192。

## Bug 3: fixedTriuMask 参数被忽略，硬编码为 false

- **位置**: 第 132-136 行
- **类型**: 功能缺陷/接口不一致
- **严重程度**: 中
- **描述**: 函数接口声明接受 `fixedTriuMask` 参数，但实现中当该参数为 `true` 时直接返回错误，且在调用内部函数时硬编码传 `false`。这意味着接口承诺的功能并未实现，对外暴露了一个无法使用的参数。如果上层框架（如 PyTorch）传入 `true`，将直接失败且错误信息中 "suppport" 拼写错误（应为 "support"）。
- **触发条件**: 调用时 `fixedTriuMask = true`。
- **测试方案**: 传入 `fixedTriuMask = true`，验证返回错误码及错误信息是否合理。

## Bug 4: 缺少输出 tensor y 的 shape 校验

- **位置**: 第 75-111 行（CheckShape 函数）
- **类型**: 校验缺失
- **严重程度**: 中
- **描述**: `CheckShape` 函数只校验了 `x` 与 `mask` 之间的 shape 关系，但未校验输出 tensor `y` 的 shape 是否与 `x` 一致。如果 `y` 的 shape 与 `x` 不匹配，在后续计算中可能导致内存越界写入。
- **触发条件**: 传入 `y` 的 shape 与 `x` 不同（如维度更小）。
- **测试方案**: 传入与 `x` shape 不一致的 `y` tensor，检查是否正确报错或产生内存越界。

## Bug 5: namespace 内声明 extern 函数位于 extern "C" 块内

- **位置**: 第 25-27 行, 第 29 行, 第 39-44 行
- **类型**: 链接/编译问题
- **严重程度**: 中
- **描述**: `extern "C"` 块（第 26 行开始）内包含了匿名 namespace，而匿名 namespace 内部又声明了 `extern` 函数 (`aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 和 `aclnnInnerScaledMaskedSoftmaxV2`)。匿名 namespace 中的符号具有内部链接属性，与 `extern` 声明矛盾，可能导致链接失败或未定义行为。同时 `extern "C"` 作用于匿名 namespace 中的 C++ 代码（使用了 `std::initializer_list` 等），存在语义冲突。
- **触发条件**: 编译链接时，如果 `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 等函数在其他编译单元定义，可能无法正确链接。
- **测试方案**: 在独立编译环境中验证链接是否成功；检查符号表中相关函数的链接属性。

## Bug 6: workspaceSize 和 executor 指针参数缺少空指针检查

- **位置**: 第 126-128 行
- **类型**: 校验缺失
- **严重程度**: 中
- **描述**: `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 接受 `uint64_t* workspaceSize` 和 `aclOpExecutor** executor` 指针参数，但未对这两个参数进行空指针校验即传递给内部函数，可能导致空指针解引用崩溃。
- **触发条件**: 调用时 `workspaceSize` 或 `executor` 为 `nullptr`。
- **测试方案**: 传入 `workspaceSize = nullptr` 或 `executor = nullptr`，验证是否触发段错误。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 描述 |
|------|------|------|----------|------|
| 1 | 115 | 逻辑错误 | 高 | 空指针校验失败时返回 ACLNN_SUCCESS 而非错误码 |
| 2 | 106 | 日志错误 | 低 | 错误日志中范围固定写 4096，未适配 8192 场景 |
| 3 | 132-136 | 功能缺陷 | 中 | fixedTriuMask=true 时直接报错，接口功能未实现且拼写错误 |
| 4 | 75-111 | 校验缺失 | 中 | 未校验输出 y 的 shape 与 x 是否一致 |
| 5 | 26-44 | 链接问题 | 中 | 匿名 namespace 内 extern 声明位于 extern "C" 块，语义冲突 |
| 6 | 126-128 | 校验缺失 | 中 | workspaceSize/executor 指针参数未做空指针检查 |
