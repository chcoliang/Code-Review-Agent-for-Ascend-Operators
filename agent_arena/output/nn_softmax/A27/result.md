# A27 代码审查报告

## Bug #1: 空指针校验失败时返回 ACLNN_SUCCESS 而非错误码

**位置**: 第 90 行，`CheckParams` 函数

**描述**: `CHECK_RET(CheckNotNull(self, out), ACLNN_SUCCESS)` 中第二个参数应为 `ACLNN_ERR_PARAM_NULLPTR`，但错误地写成了 `ACLNN_SUCCESS`。当 `self` 或 `out` 为空指针时，`CheckNotNull` 返回 false，`CHECK_RET` 宏会将 `ACLNN_SUCCESS` 作为返回值返回给调用者。这意味着空指针不会被正确报错，调用者认为参数校验通过，继续后续流程将导致空指针解引用崩溃。

**当前代码**:
```cpp
CHECK_RET(CheckNotNull(self, out), ACLNN_SUCCESS);
```

**正确代码应为**:
```cpp
CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);
```

**触发输入**:
- self: nullptr
- dim: 0
- out: 有效 tensor（shape=[2,3], dtype=FLOAT）

**预期行为**: 应返回 `ACLNN_ERR_PARAM_NULLPTR`

**实际行为**: 返回 `ACLNN_SUCCESS`，调用者误认为校验通过。随后在 `aclnnSoftmaxGetWorkspaceSize` 中继续访问 `self->IsEmpty()`（第 121 行）导致空指针解引用崩溃（segfault）。

**验证代码**:
```cpp
#include "aclnn_softmax.h"
#include <cassert>

void test_null_self_returns_success_instead_of_error() {
    // self 为 nullptr
    const aclTensor* self = nullptr;
    aclTensor* out = create_aclTensor({2, 3}, ACL_FLOAT);
    int64_t dim = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);

    // Bug: 空指针时返回 ACLNN_SUCCESS 而非 ACLNN_ERR_PARAM_NULLPTR
    // 由于 CheckParams 返回 ACLNN_SUCCESS，CHECK_RET(ret == ACLNN_SUCCESS, ret) 通过
    // 然后 self->IsEmpty() 触发 segfault
    // 如果 OP_CHECK_NULL 内部做了拦截不崩溃，则 ret == ACLNN_SUCCESS（错误）
    // 正确行为应为: assert(ret == ACLNN_ERR_PARAM_NULLPTR);
}

void test_null_out_returns_success_instead_of_error() {
    // out 为 nullptr
    aclTensor* self = create_aclTensor({2, 3}, ACL_FLOAT);
    aclTensor* out = nullptr;
    int64_t dim = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);

    // Bug: 空指针时返回 ACLNN_SUCCESS 而非 ACLNN_ERR_PARAM_NULLPTR
    // 正确行为应为: assert(ret == ACLNN_ERR_PARAM_NULLPTR);
    assert(ret == ACLNN_SUCCESS);  // 验证 bug 存在
}
```
