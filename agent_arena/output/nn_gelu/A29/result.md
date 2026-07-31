# A29 aclnn_gelu.cpp 代码审查报告

## Bug列表

### Bug #1: `out` 参数空指针未校验

- **位置**: 第31-35行，`CheckNotNull` 函数
- **描述**: `CheckNotNull` 函数中对 `out` 参数使用了 `(void)out;` 直接忽略，没有进行空指针校验。当 `out` 为 `nullptr` 时，后续在 `CheckDtypeValid`、`CheckFormat`、`CheckShape` 中访问 `out->GetDataType()`、`out->GetStorageFormat()` 等会导致空指针解引用（段错误/未定义行为）。
- **问题代码**:
  ```cpp
  static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
    OP_CHECK_NULL(self, return false);
    (void)out;  // BUG: out未做空指针校验
    return true;
  }
  ```
- **触发输入**: `self` 为有效的 float32 tensor，`out = nullptr`
- **预期异常**: 应返回 `ACLNN_ERR_PARAM_NULLPTR`，但实际会在后续函数中解引用空指针导致崩溃
- **验证代码**:
  ```cpp
  #include "aclnnop/aclnn_gelu.h"
  #include "aclnn/opdev/op_errno.h"
  #include <cassert>

  void test_out_nullptr() {
      // 构造有效的输入tensor
      int64_t shape[] = {2, 3};
      aclTensor *self = aclCreateTensor(shape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 2, nullptr);

      aclTensor *out = nullptr;  // 空指针
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      aclnnStatus ret = aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
      // 预期: 应返回 ACLNN_ERR_PARAM_NULLPTR
      // 实际: 由于未校验out, 会在后续访问out成员时崩溃
      assert(ret == ACLNN_ERR_PARAM_NULLPTR && "Should return ACLNN_ERR_PARAM_NULLPTR for null out");

      aclDestroyTensor(self);
  }
  ```
