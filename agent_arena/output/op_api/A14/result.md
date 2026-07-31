# aclnn_mul.cpp 代码审查报告

**目标平台**: Ascend 910B, CANN 8.5.0

---

## Bug #1: CheckMulPromoteType 缺少输出 dtype 转换校验

**位置**: 第 260-273 行，`CheckMulPromoteType` 函数

**类型**: 参数校验缺失

**严重程度**: Medium

**描述**: `CheckMulPromoteType` 在计算出 `promoteType` 后，无论 RegBase 还是非 RegBase 路径，都没有检查 `promoteType` 能否安全转换到 `out->GetDataType()`。对比同族函数 `CheckMulsPromoteDtype`（第 234-258 行）在 RegBase 路径中执行了 `OP_CHECK_RESULT_DTYPE_CAST_FAILED(inferDtype, out->GetDataType(), return false)`，而 `CheckInplaceMulPromoteType` 在非 RegBase 路径中也执行了 `OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, selfRef->GetDataType(), return false)`。`CheckMulPromoteType` 完全缺少此类检查，导致不兼容的输出 dtype 不会被拦截，可能产生静默数据截断/损坏。

**触发输入**:
```cpp
// self: FLOAT32 tensor [2,3]
// other: FLOAT32 tensor [2,3]
// out: INT8 tensor [2,3]  (promoteType=FLOAT32 无法安全转换到 INT8)
aclnnMulGetWorkspaceSize(self_f32, other_f32, out_i8, &workspaceSize, &executor);
```

**预期异常**: 应返回 `ACLNN_ERR_PARAM_INVALID`，提示 promoteType(FLOAT32) 无法安全转换到 out 的 dtype(INT8)。

**实际行为**: 通过参数校验，返回 `ACLNN_SUCCESS`，计算结果 Cast 到 INT8 时产生精度损失或未定义行为。

**验证代码**:
```cpp
#include "aclnn_mul.h"
#include <cassert>

void test_mul_missing_output_dtype_check() {
    // 创建两个 FLOAT32 tensor shape=[2,3]
    int64_t shape[] = {2, 3};
    auto self = aclCreateTensor(shape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 2, nullptr);
    auto other = aclCreateTensor(shape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 2, nullptr);
    // 创建 INT8 output tensor - promoteType(FLOAT32) 不应能转换到 INT8
    auto out = aclCreateTensor(shape, 2, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, shape, 2, nullptr);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    // Bug: 应返回 ACLNN_ERR_PARAM_INVALID 但实际返回 ACLNN_SUCCESS
    assert(ret == ACLNN_ERR_PARAM_INVALID);
}
```

---

## Bug #2: CheckInplaceMulShape 缺少 MAX_DIM 维度校验

**位置**: 第 300-305 行，`CheckInplaceMulShape` 函数

**类型**: 边界条件校验缺失 / 同族对称性缺陷

**严重程度**: Medium

**描述**: `CheckMulShape`（第 291-298 行）对 self 和 other 都执行了 `OP_CHECK_MAX_DIM(tensor, MAX_SUPPORT_DIMS_NUMS, return false)` 维度上限检查，而同族的 `CheckInplaceMulShape` 完全缺少此检查。超过 `MAX_SUPPORT_DIMS_NUMS`（通常为 8）维度的 tensor 在 inplace 路径中不会被拦截，可能导致后续 kernel 计算越界或未定义行为。

**触发输入**:
```cpp
// selfRef: FLOAT32 tensor 维度=9（超过 MAX_SUPPORT_DIMS_NUMS=8）
// other: FLOAT32 tensor 维度=1（可广播）
int64_t shape9d[] = {1,1,1,1,1,1,1,1,1};  // 9维
aclnnInplaceMulGetWorkspaceSize(selfRef_9d, other_1d, &workspaceSize, &executor);
```

**预期异常**: 应返回 `ACLNN_ERR_PARAM_INVALID`，提示 tensor 维度超过支持的最大值。

**实际行为**: 通过参数校验，在后续计算中可能导致内存越界或错误结果。

**验证代码**:
```cpp
#include "aclnn_mul.h"
#include <cassert>

void test_inplace_mul_missing_max_dim_check() {
    // 创建 9 维 tensor（超过 MAX_SUPPORT_DIMS_NUMS）
    int64_t shape9d[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int64_t shape1d[] = {1};
    auto selfRef = aclCreateTensor(shape9d, 9, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape9d, 9, nullptr);
    auto other = aclCreateTensor(shape1d, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape1d, 1, nullptr);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnInplaceMulGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
    // Bug: 应返回 ACLNN_ERR_PARAM_INVALID 但实际通过了校验
    assert(ret == ACLNN_ERR_PARAM_INVALID);
}
```

---

## Bug #3: CheckInplaceMulPromoteType 在 RegBase 模式下缺少输出 dtype 回转校验

**位置**: 第 275-289 行，`CheckInplaceMulPromoteType` 函数

**类型**: 参数校验路径不完整 / 同族对称性缺陷

**严重程度**: Medium

**描述**: `CheckInplaceMulPromoteType` 在 `IsRegBase()` 为 true 时（第 269 行条件），仅检查 `promoteType != DT_UNDEFINED` 就返回 true，不检查 promoteType 能否安全转换回 `selfRef->GetDataType()`。而在非 RegBase 路径中（第 285-287 行），执行了 `OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, selfRef->GetDataType(), return false)`。对于 inplace 操作，结果必须写回原 tensor，如果 promoteType 无法安全 cast 回 selfRef 的 dtype，会导致数据截断。在 Ascend 910B（RegBase 模式）上此校验被跳过。

**触发输入**:
```cpp
// selfRef: INT8 tensor [4]
// other: FLOAT32 tensor [4]
// promoteType = FLOAT32, 无法安全转换回 INT8
aclnnInplaceMulGetWorkspaceSize(selfRef_i8, other_f32, &workspaceSize, &executor);
```

**预期异常**: 应返回 `ACLNN_ERR_PARAM_INVALID`，提示 promoteType(FLOAT32) 无法安全转换为 selfRef 的 dtype(INT8)。

**实际行为**: 在 RegBase 模式（910B）下通过校验，计算结果强制 Cast 回 INT8 导致精度损失。

**验证代码**:
```cpp
#include "aclnn_mul.h"
#include <cassert>

void test_inplace_mul_regbase_missing_cast_check() {
    // 在 Ascend 910B (RegBase mode) 上运行
    int64_t shape[] = {4};
    auto selfRef = aclCreateTensor(shape, 1, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, shape, 1, nullptr);
    auto other = aclCreateTensor(shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 1, nullptr);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnInplaceMulGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
    // Bug: 在 RegBase 模式下应返回 ACLNN_ERR_PARAM_INVALID 但实际返回 ACLNN_SUCCESS
    assert(ret == ACLNN_ERR_PARAM_INVALID);
}
```

---

## 审查总结

| # | 函数 | Bug 类型 | 严重程度 |
|---|------|----------|----------|
| 1 | CheckMulPromoteType | 输出 dtype 转换校验缺失 | Medium |
| 2 | CheckInplaceMulShape | MAX_DIM 边界校验缺失 | Medium |
| 3 | CheckInplaceMulPromoteType | RegBase 路径 dtype 回转校验缺失 | Medium |

三个 bug 均为同族函数间的对称性缺陷——其他变体（Muls/非 RegBase 路径）有对应的校验，但这些路径遗漏了。
