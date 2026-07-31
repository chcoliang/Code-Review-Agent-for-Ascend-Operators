# A28 代码审查报告

## Bug #1: CheckShape 缺少维度上限校验（OP_CHECK_MAX_DIM）

**位置**: 第 80-84 行，`CheckShape` 函数

**描述**: `CheckShape` 函数中缺少 `OP_CHECK_MAX_DIM(self, AXIS_LIMIT, return false)` 校验。虽然定义了 `AXIS_LIMIT = 8`（第 47 行），但 `CheckShape` 中没有使用它来检查输入 tensor 的维度是否超过 8 维。底层 SoftmaxV2 算子不支持超过 8 维的 tensor，缺少此校验会导致超过 8 维的 tensor 绕过参数检查，在底层算子执行时可能产生未定义行为、计算错误或崩溃。

**当前代码**:
```cpp
static bool CheckShape(const aclTensor *self, aclTensor *out) {
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}
```

**正确代码应为**:
```cpp
static bool CheckShape(const aclTensor *self, aclTensor *out) {
    OP_CHECK_MAX_DIM(self, AXIS_LIMIT, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}
```

**触发输入**:
- self: dtype=FLOAT, shape=[2,2,2,2,2,2,2,2,2] (9维 tensor)
- dim: 0
- out: dtype=FLOAT, shape=[2,2,2,2,2,2,2,2,2] (9维 tensor)

**预期行为**: 应返回 `ACLNN_ERR_PARAM_INVALID`（超过 8 维限制）

**实际行为**: 返回 `ACLNN_SUCCESS`，通过参数校验，在底层 SoftmaxV2 执行时可能崩溃或产生错误结果

**验证代码**:
```cpp
#include "aclnn_softmax.h"
#include <cassert>

void test_9d_tensor_should_be_rejected() {
    // 创建 9 维 tensor，超过 AXIS_LIMIT=8 的限制
    aclTensor* self = create_aclTensor({2,2,2,2,2,2,2,2,2}, ACL_FLOAT);  // 9维
    aclTensor* out = create_aclTensor({2,2,2,2,2,2,2,2,2}, ACL_FLOAT);   // 9维
    int64_t dim = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);

    // Bug: 缺少维度上限检查，9维 tensor 不会被拒绝
    // 实际返回 ACLNN_SUCCESS（错误地通过了校验）
    assert(ret == ACLNN_SUCCESS);  // 验证 bug 存在
    // 正确行为应为: assert(ret == ACLNN_ERR_PARAM_INVALID);
}

void test_8d_tensor_should_pass() {
    // 创建 8 维 tensor，恰好在限制范围内
    aclTensor* self = create_aclTensor({2,2,2,2,2,2,2,2}, ACL_FLOAT);  // 8维
    aclTensor* out = create_aclTensor({2,2,2,2,2,2,2,2}, ACL_FLOAT);   // 8维
    int64_t dim = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);

    // 8维应该通过
    assert(ret == ACLNN_SUCCESS);
}
```
