#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_gelu.h"
#include <cstdio>
#include <cstdlib>

// Bug #1: 输出张量数据类型未校验
// self: FP16, out: INT32 (不支持的 dtype)
// 预期返回 ACLNN_ERR_PARAM_INVALID，实际可能通过校验导致数据损坏
int main() {
    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", initRet);
        return 1;
    }

    int32_t deviceId = 0;
    aclError setRet = aclrtSetDevice(deviceId);
    if (setRet != ACL_SUCCESS) {
        printf("aclrtSetDevice failed: %d\n", setRet);
        aclFinalize();
        return 1;
    }

    // self: shape=[2,3], dtype=FP16
    int64_t shape[] = {2, 3};
    int64_t strides[] = {3, 1};
    auto self = aclCreateTensor(shape, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);

    // out: shape=[2,3], dtype=INT32 (unsupported for GeLU output)
    auto out = aclCreateTensor(shape, 2, ACL_INT32, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);

    if (self == nullptr || out == nullptr) {
        printf("Failed to create tensors\n");
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Calling aclnnGeluGetWorkspaceSize with out dtype=INT32...\n");
    aclError ret = (aclError)aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
    printf("Return code: %d\n", (int)ret);

    if (ret != ACL_SUCCESS) {
        printf("PASS: Got error code as expected (invalid out dtype rejected)\n");
    } else {
        printf("FAIL: Should have returned error for unsupported out dtype (Bug #1 confirmed)\n");
    }

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
