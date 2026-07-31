# A25 代码审查报告

## Bug #1: ASCEND910B 数据类型支持列表缺少 BF16

**位置**: 第 44-45 行，`ASCEND910B_DTYPE_SUPPORT_LIST` 定义

**描述**: 根据代码注释（第 40 行）明确说明 "AIC支持: DT_BF16, DT_FLOAT16, DT_FLOAT"，但 `ASCEND910B_DTYPE_SUPPORT_LIST` 中未包含 `DT_BF16`。在 Ascend 910B 平台上，BF16 类型的输入 tensor 会被错误拒绝，返回 `ACLNN_ERR_PARAM_INVALID`，而实际上硬件是支持该类型的。

**当前代码**:
```cpp
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};
```

**触发输入**:
- 平台: Ascend 910B
- self: dtype=BF16, shape=[2,3]
- dim: 1
- out: dtype=BF16, shape=[2,3]

**预期行为**: 应返回 `ACLNN_SUCCESS`（BF16 在 910B 上是支持的）

**实际行为**: 返回 `ACLNN_ERR_PARAM_INVALID`

**验证代码**:
```cpp
#include "aclnn_softmax.h"
#include <cassert>

void test_bf16_rejected_on_910b() {
    // 在 Ascend 910B 平台上运行
    // 创建 BF16 类型的输入和输出 tensor
    aclTensor* self = create_aclTensor({2, 3}, ACL_BF16);  // shape=[2,3], dtype=BF16
    aclTensor* out = create_aclTensor({2, 3}, ACL_BF16);   // shape=[2,3], dtype=BF16
    int64_t dim = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);

    // Bug: 910B 硬件支持 BF16，但代码会拒绝
    // 预期应返回 ACLNN_SUCCESS，实际返回 ACLNN_ERR_PARAM_INVALID
    assert(ret == ACLNN_ERR_PARAM_INVALID);  // 验证 bug 存在
    // 正确行为应为: assert(ret == ACLNN_SUCCESS);
}
```
