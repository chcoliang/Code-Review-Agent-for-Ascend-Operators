# Code Review: A122 - aclnn_scaled_masked_softmax.cpp

## Bug 1: 缺少输出张量 y 的空指针校验

- **位置**: 第 46-52 行 (`CheckNotNull` 函数)
- **类型**: 参数校验缺失
- **严重程度**: 高
- **描述**: `CheckNotNull` 函数接收参数 `aclTensor* y`，但函数体中只对 `x` 和 `mask` 进行了 `OP_CHECK_NULL` 校验，缺少对 `y` 的空指针检查。当用户传入 `y = nullptr` 时，后续对 `y` 的 dtype 检查（第 58 行）、维度检查（第 69 行）等操作将导致空指针解引用，引发段错误（segfault）或未定义行为。
- **触发条件**: 调用 `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 时传入 `y = nullptr`。
- **修复建议**: 在第 49 行之后添加 `OP_CHECK_NULL(y, return false);`
- **测试方案**:
  1. 构造合法的 `x` 和 `mask` 张量，传入 `y = nullptr`，验证函数返回 `ACLNN_ERR_INNER_NULLPTR` 而非崩溃。
  2. 在 ASAN（AddressSanitizer）模式下运行，确认不存在空指针解引用。

---

## 汇总表

| 编号 | 行号 | Bug 类型 | 严重程度 | 简述 |
|------|------|----------|----------|------|
| 1 | 46-52 | 参数校验缺失 | 高 | 缺少对输出张量 `y` 的空指针检查，可能导致段错误 |
