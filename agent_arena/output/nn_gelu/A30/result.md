# A30 aclnn_gelu.cpp 代码审查报告

## Bug列表

### Bug #1: DTYPE_SUPPORT_LIST 中错误地包含了 DT_INT32

- **位置**: 第22-24行，`DTYPE_SUPPORT_LIST` 定义
- **描述**: GELU 是浮点激活函数（`GELU(x) = x * Φ(x)`），其数学定义依赖于连续函数的性质，不支持整数类型。`DTYPE_SUPPORT_LIST` 中错误地包含了 `DataType::DT_INT32`。当传入 INT32 类型的 tensor 时，数据类型校验会通过，但底层 Gelu l0 算子不支持 INT32 运算，会导致计算错误或内部异常。
- **问题代码**:
  ```cpp
  static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT32  // BUG: DT_INT32不应在此
  };
  ```
- **触发输入**: `self` 和 `out` 均为 INT32 类型、shape 为 `[2, 3]` 的 tensor
- **预期异常**: 应返回 `ACLNN_ERR_PARAM_INVALID`（数据类型不支持），但实际会通过校验进入计算流程导致未定义行为
- **验证代码**:
  ```cpp
  #include "aclnnop/aclnn_gelu.h"
  #include "aclnn/opdev/op_errno.h"
  #include <cassert>

  void test_int32_dtype_should_reject() {
      // 构造INT32类型的tensor
      int64_t shape[] = {2, 3};
      aclTensor *self = aclCreateTensor(shape, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, shape, 2, nullptr);
      aclTensor *out = aclCreateTensor(shape, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, shape, 2, nullptr);

      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      aclnnStatus ret = aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
      // 预期: 应返回 ACLNN_ERR_PARAM_INVALID, 因为GELU不支持INT32
      // 实际: 由于DT_INT32在支持列表中, 校验通过, 进入l0op::Gelu可能产生异常
      assert(ret == ACLNN_ERR_PARAM_INVALID && "INT32 should not be supported by GELU");

      aclDestroyTensor(self);
      aclDestroyTensor(out);
  }
  ```
