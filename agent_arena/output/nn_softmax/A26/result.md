# A26 代码审查报告

## Bug #1: ASCEND910B 数据类型支持列表错误包含 DT_INT32

**位置**: 第 44-45 行，`ASCEND910B_DTYPE_SUPPORT_LIST` 定义

**描述**: `ASCEND910B_DTYPE_SUPPORT_LIST` 中包含了 `op::DataType::DT_INT32`，这是一个整数类型。Softmax 是浮点运算（涉及指数函数和除法），对整数类型没有数学意义，且底层 SoftmaxV2 算子不支持整数类型输入。传入 INT32 类型的 tensor 会通过参数校验，但在后续算子执行时可能产生未定义行为、计算错误或崩溃。

**当前代码**:
```cpp
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16, op::DataType::DT_INT32};
```

**触发输入**:
- 平台: Ascend 910B
- self: dtype=INT32, shape=[2,3], data=[[1,2,3],[4,5,6]]
- dim: 1
- out: dtype=INT32, shape=[2,3]

**预期行为**: 应返回 `ACLNN_ERR_PARAM_INVALID`（INT32 不应支持 softmax 运算）

**实际行为**: 返回 `ACLNN_SUCCESS`，通过参数校验，后续执行时可能产生错误结果或崩溃

**验证代码**:
```cpp
#include "aclnn_softmax.h"
#include <cassert>

void test_int32_should_be_rejected_on_910b() {
    // 在 Ascend 910B 平台上运行
    // 创建 INT32 类型的输入和输出 tensor
    aclTensor* self = create_aclTensor({2, 3}, ACL_INT32);  // shape=[2,3], dtype=INT32
    aclTensor* out = create_aclTensor({2, 3}, ACL_INT32);   // shape=[2,3], dtype=INT32
    int64_t dim = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);

    // Bug: INT32 不应该被 softmax 支持，但代码允许其通过
    // 实际返回 ACLNN_SUCCESS（通过了校验），这是错误的
    assert(ret == ACLNN_SUCCESS);  // 验证 bug 存在：不应该通过
    // 正确行为应为: assert(ret == ACLNN_ERR_PARAM_INVALID);
}
```
