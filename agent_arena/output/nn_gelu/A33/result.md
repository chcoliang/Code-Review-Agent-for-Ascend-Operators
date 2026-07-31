# A33 aclnn_gelu.cpp 代码审查报告

## Bug列表

### Bug #1: `CheckShape` 函数未校验 self 和 out 的 shape 一致性

- **位置**: 第51-55行，`CheckShape` 函数
- **描述**: `CheckShape` 函数中对 `out` 参数使用了 `(void)out;` 直接忽略，缺少 `OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);` 校验。GELU 算子要求输出 tensor 的 shape 必须与输入 tensor 的 shape 完全一致。当 `self` 和 `out` 的 shape 不同时（如 self 为 `[2,3]`，out 为 `[3,2]`），校验会通过，后续 `l0op::ViewCopy` 将计算结果拷贝到 shape 不匹配的 out tensor 中，导致数据越界访问或计算结果错误。
- **问题代码**:
  ```cpp
  static bool CheckShape(const aclTensor *self, const aclTensor *out) {
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    (void)out;  // BUG: 缺少shape一致性校验
    return true;
  }
  ```
- **触发输入**: `self` 为 shape `[2, 3]` 的 float32 tensor，`out` 为 shape `[6]` 或 `[3, 2]` 的 float32 tensor
- **预期异常**: 应返回 `ACLNN_ERR_PARAM_INVALID`（shape不匹配）
- **实际行为**: 校验通过，进入计算流程，ViewCopy 操作可能导致内存越界或结果写入错误位置
- **验证代码**:
  ```cpp
  #include "aclnnop/aclnn_gelu.h"
  #include "aclnn/opdev/op_errno.h"
  #include <cassert>

  void test_shape_mismatch_not_detected() {
      // self shape [2, 3], out shape [3, 2] - shape不一致
      int64_t self_shape[] = {2, 3};
      int64_t out_shape[] = {3, 2};
      aclTensor *self = aclCreateTensor(self_shape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, self_shape, 2, nullptr);
      aclTensor *out = aclCreateTensor(out_shape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, out_shape, 2, nullptr);

      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      aclnnStatus ret = aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
      // 预期: 应返回 ACLNN_ERR_PARAM_INVALID, 因为self和out的shape不一致
      // 实际: 由于CheckShape未校验out, 返回ACLNN_SUCCESS, 后续ViewCopy行为未定义
      assert(ret == ACLNN_ERR_PARAM_INVALID && "Shape mismatch should be rejected");

      aclDestroyTensor(self);
      aclDestroyTensor(out);
  }

  void test_different_ndim_not_detected() {
      // self shape [2, 3], out shape [6] - 维度数不同
      int64_t self_shape[] = {2, 3};
      int64_t out_shape[] = {6};
      aclTensor *self = aclCreateTensor(self_shape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, self_shape, 2, nullptr);
      aclTensor *out = aclCreateTensor(out_shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, out_shape, 1, nullptr);

      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      aclnnStatus ret = aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
      // 预期: 应返回 ACLNN_ERR_PARAM_INVALID
      // 实际: 校验通过, 可能导致内存错误
      assert(ret == ACLNN_ERR_PARAM_INVALID && "Different ndim should be rejected");

      aclDestroyTensor(self);
      aclDestroyTensor(out);
  }
  ```
