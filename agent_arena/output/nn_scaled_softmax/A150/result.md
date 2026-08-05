# Code Review: aclnn_scaled_masked_softmax.cpp

## Bug List

### Bug 1: 缺少对输入张量 x 的数据类型直接校验

- **位置**: 第 55-63 行, `CheckDtypeValid` 函数
- **类型**: 输入校验缺失
- **严重程度**: 中
- **描述**: 函数仅对 `mask` 和 `y` 做了 `OP_CHECK_DTYPE_NOT_SUPPORT` 校验，对 `x` 未直接校验其 dtype 是否在 `SOFTMAX_X_DTYPE_SUPPORT_LIST` 中。虽然通过 `OP_CHECK_DTYPE_NOT_SAME(x, y)` 间接约束，但当 x 的 dtype 不合法时，报错信息会指向 y 或指向 "x 与 y 不一致"，误导用户。且如果后续维护者移除 SAME 校验，x 的类型将完全不受限。
- **触发条件**: 传入 `x` 的 dtype 为 DT_INT32 等不支持类型，而 `y` 的 dtype 为 DT_FLOAT16（合法类型），此时报错为 "x 和 y dtype 不一致" 而非 "x dtype 不支持"。
- **测试方案**: 构造 x(DT_INT32), mask(DT_BOOL), y(DT_FLOAT16) 调用接口，验证返回的错误码和错误信息是否准确指出 x 的 dtype 不支持。

---

### Bug 2: 错误信息与实际限制值不匹配

- **位置**: 第 105 行
- **类型**: 错误信息不准确
- **严重程度**: 低
- **描述**: 当平台为 ASCEND910_95 时，`dDimLimit` 被设为 8192，但错误日志仍硬编码输出 `"Expected x and mask dim4 in range of (0, 4096]."`, 未反映实际限制。这会误导用户以为上限是 4096，实际上限为 8192。
- **触发条件**: 在 ASCEND910_95 平台上，传入 dim3 值为 5000（大于 4096 但小于 8192），此时不会报错，但如果传入 9000 触发报错，提示信息却显示上限为 4096。
- **测试方案**: 在 ASCEND910_95 平台上传入 dim3=9000 的张量，检查报错信息是否正确显示上限为 8192。

---

### Bug 3: 匿名命名空间内的 extern 函数声明导致链接问题

- **位置**: 第 39-44 行
- **类型**: 编译/链接错误
- **严重程度**: 高
- **描述**: `extern` 函数声明 `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 和 `aclnnInnerScaledMaskedSoftmaxV2` 被放置在匿名 namespace 内。匿名命名空间赋予内部链接(internal linkage)，而 `extern` 声明请求外部链接(external linkage)，两者矛盾。根据 C++ 标准，这可能导致链接器找不到符号定义，或在某些编译器上产生未定义行为。
- **触发条件**: 使用严格遵循标准的编译器(如较新版本 GCC/Clang 的 `-std=c++17` 模式)编译该文件时，链接阶段可能报 undefined reference 错误。
- **测试方案**: 使用 `-std=c++17 -pedantic` 编译选项编译该文件，观察是否有链接错误或编译警告。

---

### Bug 4: 缺少输出张量 y 与输入张量 x 的 shape 一致性校验

- **位置**: 第 74-110 行, `CheckShape` 函数 及 第 112-122 行, `CheckParams` 函数
- **类型**: 输入校验缺失
- **严重程度**: 高
- **描述**: `CheckShape` 只校验了 x 和 mask 的 shape 关系，但从未校验输出张量 `y` 的 shape 是否与 `x` 一致。Softmax 操作要求输出与输入同 shape，如果 `y` 的 shape 不匹配，内核写入时会发生越界访问或计算结果错误。
- **触发条件**: 传入 y 的 shape 与 x 不一致（如 x 为 [2,4,8,64]，y 为 [2,4,8,32]），校验通过后执行内核导致内存越界。
- **测试方案**: 构造 x shape=[2,4,8,64], mask shape=[2,4,8,64], y shape=[2,4,8,32]，调用 GetWorkspaceSize 接口，验证是否返回参数错误。

---

### Bug 5: 缺少对 workspaceSize 和 executor 指针参数的空指针校验

- **位置**: 第 125-137 行, `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 函数
- **类型**: 空指针解引用风险
- **严重程度**: 中
- **描述**: 外部 API 函数接收 `uint64_t* workspaceSize` 和 `aclOpExecutor** executor` 指针参数，但在传递给内部函数 `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 之前未做空指针检查。如果调用者传入 nullptr，将导致段错误。
- **触发条件**: 调用者传入 `workspaceSize = nullptr` 或 `executor = nullptr`。
- **测试方案**: 分别传入 nullptr 给 workspaceSize 和 executor 参数，验证函数是否安全返回错误码而非崩溃。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 55-63 行 | 输入校验缺失 | 中 | 未直接校验 x 的 dtype，报错信息误导 |
| 2 | 第 105 行 | 错误信息不准确 | 低 | 错误提示硬编码 4096，未适配 8192 场景 |
| 3 | 第 39-44 行 | 编译/链接错误 | 高 | 匿名 namespace 内 extern 声明导致链接冲突 |
| 4 | 第 74-122 行 | 输入校验缺失 | 高 | 缺少输出 y 与输入 x 的 shape 一致性校验 |
| 5 | 第 125-137 行 | 空指针解引用风险 | 中 | 未校验 workspaceSize/executor 指针是否为空 |
