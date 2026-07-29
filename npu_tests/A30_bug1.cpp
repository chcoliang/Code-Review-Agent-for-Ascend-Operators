#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_gelu.h"
#include <cassert>
#include <cstdio>
#include <cstdint>

// Bug: DTYPE_SUPPORT_LIST 中错误包含 DT_INT32
// GELU是浮点激活函数，不支持整数类型
void test_int32_dtype_should_reject() {
    int64_t shape[] = {2, 3};
    int64_t strides[] = {3, 1};
    aclTensor *self = aclCreateTensor(shape, 2, ACL_INT32, strides, 0,
                                       ACL_FORMAT_ND, shape, 2, nullptr);
    aclTensor *out = aclCreateTensor(shape, 2, ACL_INT32, strides, 0,
                                      ACL_FORMAT_ND, shape, 2, nullptr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor));

    // 预期: 应返回 ACLNN_ERR_PARAM_INVALID, 因为GELU不支持INT32
    // 实际: 由于DT_INT32在支持列表中, 校验通过
    printf("A30 Bug1 test_int32_dtype: ret = %d (expected ACLNN_ERR_PARAM_INVALID)\n", (int)ret);

    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
}

int main() {
    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", (int)initRet);
        return -1;
    }

    test_int32_dtype_should_reject();

    aclFinalize();
    return 0;
}
