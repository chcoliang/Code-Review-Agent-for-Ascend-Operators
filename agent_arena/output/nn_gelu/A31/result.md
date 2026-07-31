# A31 aclnn_gelu.cpp 代码审查报告

## Bug列表

### Bug #1: 空指针校验失败时返回错误码为 ACLNN_SUCCESS

- **位置**: 第70行，`CheckParams` 函数
- **描述**: `CHECK_RET(CheckNotNull(self, out), ACLNN_SUCCESS);` 中第二个参数应为 `ACLNN_ERR_PARAM_NULLPTR`，但错误地写成了 `ACLNN_SUCCESS`。当 `self` 或 `out` 为 `nullptr` 时，`CheckNotNull` 返回 `false`，`CHECK_RET` 宏会返回 `ACLNN_SUCCESS`，调用方会认为参数校验通过，继续执行后续逻辑，导致空指针解引用崩溃。
- **问题代码**:
  ```cpp
  static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_SUCCESS);  // BUG: 应为 ACLNN_ERR_PARAM_NULLPTR
    ...
  }
  ```
- **触发输入**: `self = nullptr`，`out` 为有效 tensor
- **预期异常**: 应返回 `ACLNN_ERR_PARAM_NULLPTR`
- **实际行为**: 返回 `ACLNN_SUCCESS`，随后 `aclnnGeluGetWorkspaceSize` 中 `ret == ACLNN_SUCCESS` 校验通过，继续执行 `self->IsEmpty()` 导致空指针解引用
- **验证代码**:
  ```cpp
  #include "aclnnop/aclnn_gelu.h"
  #include "aclnn/opdev/op_errno.h"
  #include <cassert>

  void test_null_self_returns_wrong_status() {
      // self为空指针
      aclTensor *self = nullptr;
      int64_t shape[] = {2, 3};
      aclTensor *out = aclCreateTensor(shape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 2, nullptr);

      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      aclnnStatus ret = aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
      // 预期: 应返回 ACLNN_ERR_PARAM_NULLPTR
      // 实际: 返回 ACLNN_SUCCESS (因为错误码写错), 然后在self->IsEmpty()处崩溃
      assert(ret == ACLNN_ERR_PARAM_NULLPTR && "Should return ACLNN_ERR_PARAM_NULLPTR for null self");

      aclDestroyTensor(out);
  }

  void test_null_out_returns_wrong_status() {
      int64_t shape[] = {2, 3};
      aclTensor *self = aclCreateTensor(shape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 2, nullptr);
      aclTensor *out = nullptr;

      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      aclnnStatus ret = aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
      // 预期: 应返回 ACLNN_ERR_PARAM_NULLPTR
      // 实际: 返回 ACLNN_SUCCESS, 然后在CheckDtypeValid中访问out->GetDataType()崩溃
      assert(ret == ACLNN_ERR_PARAM_NULLPTR && "Should return ACLNN_ERR_PARAM_NULLPTR for null out");

      aclDestroyTensor(self);
  }
  ```
