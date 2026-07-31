# A32 aclnn_gelu.cpp 代码审查报告

## Bug列表

### Bug #1: 缺少空 tensor 处理逻辑

- **位置**: 第94-96行，`aclnnGeluGetWorkspaceSize` 函数中参数检查之后
- **描述**: 函数在参数校验通过后，直接进入 `l0op::Contiguous` 调用，缺少对空 tensor（元素个数为0）的提前返回处理。当 `self` 是空 tensor（如 shape 为 `[0]` 或 `[2, 0, 3]`）时，不需要进行实际计算，应直接设置 `workspaceSize = 0` 并返回。缺少此逻辑可能导致后续 l0 算子处理空 tensor 时产生未定义行为或不必要的资源分配。
- **问题代码**:
  ```cpp
  // 固定写法，参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 缺少: 空tensor处理
  // if (self->IsEmpty()) {
  //   *workspaceSize = 0;
  //   uniqueExecutor.ReleaseTo(executor);
  //   return ACLNN_SUCCESS;
  // }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  ```
- **触发输入**: `self` 和 `out` 均为 shape `[0]` 或 `[2, 0, 3]` 的 float32 tensor（空 tensor）
- **预期行为**: 应直接返回 `ACLNN_SUCCESS` 且 `workspaceSize = 0`，不进入计算流程
- **实际行为**: 空 tensor 被传入 `l0op::Contiguous` 和 `l0op::Gelu`，可能导致内部异常或未定义行为
- **验证代码**:
  ```cpp
  #include "aclnnop/aclnn_gelu.h"
  #include "aclnn/opdev/op_errno.h"
  #include <cassert>

  void test_empty_tensor_handling() {
      // 构造空tensor (shape中有0维)
      int64_t shape[] = {2, 0, 3};
      aclTensor *self = aclCreateTensor(shape, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 3, nullptr);
      aclTensor *out = aclCreateTensor(shape, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 3, nullptr);

      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      aclnnStatus ret = aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
      // 预期: 应返回 ACLNN_SUCCESS 且 workspaceSize == 0, 不进入实际计算
      // 实际: 空tensor被传入l0op算子, 可能导致异常
      assert(ret == ACLNN_SUCCESS && "Empty tensor should be handled gracefully");
      assert(workspaceSize == 0 && "Empty tensor should require no workspace");

      aclDestroyTensor(self);
      aclDestroyTensor(out);
  }

  void test_zero_dim_tensor() {
      // shape为[0]的tensor
      int64_t shape[] = {0};
      aclTensor *self = aclCreateTensor(shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, shape, 1, nullptr);
      aclTensor *out = aclCreateTensor(shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, shape, 1, nullptr);

      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      aclnnStatus ret = aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
      assert(ret == ACLNN_SUCCESS && "Zero-element tensor should succeed");
      assert(workspaceSize == 0 && "Zero-element tensor needs no workspace");

      aclDestroyTensor(self);
      aclDestroyTensor(out);
  }
  ```
