# Code Review Result - A119

**文件**: `aclnn_scaled_masked_softmax.cpp`
**审查对象**: Ascend 910B ScaledMaskedSoftmax op_api 层

---

## Bug #1: 输出张量 y 空指针校验缺失

| 属性 | 内容 |
|------|------|
| **位置** | 第 46-52 行，`CheckNotNull` 函数 |
| **类型** | 参数校验缺失（空指针） |
| **严重程度** | 高 |

### 描述

`CheckNotNull` 函数仅校验了 `x` 和 `mask` 的空指针，但遗漏了对输出张量 `y` 的空指针校验。函数签名接收 `aclTensor* y` 参数，但函数体中缺少 `OP_CHECK_NULL(y, return false);`。

当 `y` 为 `nullptr` 时，后续流程（如 `CheckDtypeValid`、`CheckTensorDim`）会对 `y` 解引用，导致段错误（segfault）或未定义行为。

### 问题代码

```cpp
static bool CheckNotNull(const aclTensor* x, const aclTensor* mask, aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(mask, return false);
    // 缺少: OP_CHECK_NULL(y, return false);
    return true;
}
```

### 修复建议

```cpp
static bool CheckNotNull(const aclTensor* x, const aclTensor* mask, aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(mask, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}
```

### 触发条件

调用 `aclnnScaledMaskedSoftmaxGetWorkspaceSize(x, mask, scale, fixedTriuMask, nullptr, &workspaceSize, &executor)` 时，`y` 传入 `nullptr`。

### 测试方案

```cpp
TEST(ScaledMaskedSoftmax, NullOutputTensor) {
    aclTensor* x = createValidTensor({2, 8, 64, 128}, DT_FLOAT16);
    aclTensor* mask = createValidTensor({2, 8, 64, 128}, DT_BOOL);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // y 传入 nullptr，应返回 ACLNN_ERR_INNER_NULLPTR 而非崩溃
    auto ret = aclnnScaledMaskedSoftmaxGetWorkspaceSize(
        x, mask, 1.0, false, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_INNER_NULLPTR);
}
```

---

## 汇总表

| # | 行号 | 类型 | 严重程度 | 简述 |
|---|------|------|----------|------|
| 1 | 46-52 | 空指针校验缺失 | 高 | 输出张量 `y` 未进行空指针检查，可导致段错误 |
