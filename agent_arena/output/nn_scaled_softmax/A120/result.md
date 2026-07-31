# Code Review Result - A120

**文件**: `aclnn_scaled_masked_softmax.cpp`
**审查对象**: Ascend 910B ScaledMaskedSoftmax op_api 层

---

## Bug #1: dtype 白名单遗漏 BF16 类型

| 属性 | 内容 |
|------|------|
| **位置** | 第 34-35 行，`SOFTMAX_X_DTYPE_SUPPORT_LIST` 定义 |
| **类型** | dtype 白名单遗漏 |
| **严重程度** | 高 |

### 描述

`SOFTMAX_X_DTYPE_SUPPORT_LIST` 仅包含 `{DT_FLOAT, DT_FLOAT16}`，缺少 `DT_BF16`。Ascend 910B 硬件原生支持 BF16 运算，ScaledMaskedSoftmax 算子在大模型推理场景中广泛使用 BF16 精度。遗漏 BF16 将导致 BF16 输入被错误拒绝，返回 `ACLNN_ERR_PARAM_INVALID`，导致依赖 BF16 的 LLM 推理流水线失败。

### 问题代码

```cpp
static const std::initializer_list<op::DataType> SOFTMAX_X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
```

### 修复建议

```cpp
static const std::initializer_list<op::DataType> SOFTMAX_X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
```

### 触发条件

当用户传入 BF16 类型的输入张量 `x` 和输出张量 `y` 时，`CheckDtypeValid` 中的 `OP_CHECK_DTYPE_NOT_SUPPORT` 会拒绝合法的 BF16 输入。

### 测试方案

```cpp
TEST(ScaledMaskedSoftmax, BF16TypeSupport) {
    aclTensor* x = createValidTensor({2, 8, 64, 128}, DT_BF16);
    aclTensor* mask = createValidTensor({2, 8, 64, 128}, DT_BOOL);
    aclTensor* y = createValidTensor({2, 8, 64, 128}, DT_BF16);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // BF16 应被支持，不应返回错误
    auto ret = aclnnScaledMaskedSoftmaxGetWorkspaceSize(
        x, mask, 1.0, false, y, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST(ScaledMaskedSoftmax, FP32TypeStillSupported) {
    aclTensor* x = createValidTensor({2, 8, 64, 128}, DT_FLOAT);
    aclTensor* mask = createValidTensor({2, 8, 64, 128}, DT_BOOL);
    aclTensor* y = createValidTensor({2, 8, 64, 128}, DT_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnScaledMaskedSoftmaxGetWorkspaceSize(
        x, mask, 1.0, false, y, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}
```

---

## 汇总表

| # | 行号 | 类型 | 严重程度 | 简述 |
|---|------|------|----------|------|
| 1 | 34-35 | dtype 白名单遗漏 | 高 | `SOFTMAX_X_DTYPE_SUPPORT_LIST` 缺少 `DT_BF16`，导致合法 BF16 输入被拒绝 |
