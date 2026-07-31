# Code Review Result - A121

**文件**: `aclnn_scaled_masked_softmax.cpp`
**审查对象**: Ascend 910B ScaledMaskedSoftmax op_api 层

---

## Bug #1: dtype 白名单过宽 - 错误地包含 DT_INT32

| 属性 | 内容 |
|------|------|
| **位置** | 第 34-35 行，`SOFTMAX_X_DTYPE_SUPPORT_LIST` 定义 |
| **类型** | dtype 白名单过宽 |
| **严重程度** | 高 |

### 描述

`SOFTMAX_X_DTYPE_SUPPORT_LIST` 中错误地包含了 `op::DataType::DT_INT32`。Softmax 算子本质是浮点运算（涉及 exp、求和、除法），对整型数据进行 softmax 在数学上无意义且会导致：

1. **精度灾难**：整型无法表示 softmax 输出的 (0,1) 区间概率值，结果全部截断为 0 或 1
2. **内核崩溃/未定义行为**：底层 Ascend kernel（ScaledMaskedSoftmaxV2）的 tiling 和计算逻辑未针对 INT32 设计，可能导致非法内存访问或硬件异常
3. **静默错误**：用户误传 INT32 张量时不会收到错误提示，而是得到完全错误的计算结果

### 问题代码

```cpp
static const std::initializer_list<op::DataType> SOFTMAX_X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT32};
```

### 修复建议

```cpp
static const std::initializer_list<op::DataType> SOFTMAX_X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
```

### 触发条件

当用户传入 `DT_INT32` 类型的张量作为输入 `x` 和输出 `y` 时，参数校验不会拦截，数据将被传递给底层 kernel 进行非法计算。

### 测试方案

```cpp
TEST(ScaledMaskedSoftmax, INT32TypeShouldBeRejected) {
    aclTensor* x = createValidTensor({2, 8, 64, 128}, DT_INT32);
    aclTensor* mask = createValidTensor({2, 8, 64, 128}, DT_BOOL);
    aclTensor* y = createValidTensor({2, 8, 64, 128}, DT_INT32);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // INT32 不应被支持，应返回参数错误
    auto ret = aclnnScaledMaskedSoftmaxGetWorkspaceSize(
        x, mask, 1.0, false, y, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST(ScaledMaskedSoftmax, ValidFloatTypesStillAccepted) {
    // 确认 FP32/FP16/BF16 仍然正常工作
    for (auto dtype : {DT_FLOAT, DT_FLOAT16, DT_BF16}) {
        aclTensor* x = createValidTensor({2, 8, 64, 128}, dtype);
        aclTensor* mask = createValidTensor({2, 8, 64, 128}, DT_BOOL);
        aclTensor* y = createValidTensor({2, 8, 64, 128}, dtype);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;

        auto ret = aclnnScaledMaskedSoftmaxGetWorkspaceSize(
            x, mask, 1.0, false, y, &workspaceSize, &executor);
        EXPECT_EQ(ret, ACLNN_SUCCESS);
    }
}
```

---

## 汇总表

| # | 行号 | 类型 | 严重程度 | 简述 |
|---|------|------|----------|------|
| 1 | 34-35 | dtype 白名单过宽 | 高 | `SOFTMAX_X_DTYPE_SUPPORT_LIST` 错误包含 `DT_INT32`，允许非法整型输入通过校验 |
