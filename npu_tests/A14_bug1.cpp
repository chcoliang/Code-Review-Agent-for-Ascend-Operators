#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_mul.h"
#include <cstdio>
#include <cstdlib>

// Bug #1: CheckMulPromoteType 缺少输出 dtype 转换校验
// self: FLOAT32, other: FLOAT32, out: INT8
// promoteType=FLOAT32 无法安全转换到 INT8
// 预期返回 ACLNN_ERR_PARAM_INVALID，实际返回 ACLNN_SUCCESS
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

    // self: FLOAT32 tensor shape=[2,3]
    int64_t shape[] = {2, 3};
    int64_t strides[] = {3, 1};
    auto self = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);

    // other: FLOAT32 tensor shape=[2,3]
    auto other = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);

    // out: INT8 tensor shape=[2,3] (promoteType=FLOAT32 cannot safely cast to INT8)
    auto out = aclCreateTensor(shape, 2, ACL_INT8, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);

    if (self == nullptr || other == nullptr || out == nullptr) {
        printf("Failed to create tensors\n");
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Calling aclnnMulGetWorkspaceSize with out dtype=INT8 (incompatible with promoteType=FLOAT32)...\n");
    aclError ret = (aclError)aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    printf("Return code: %d\n", (int)ret);

    if (ret != ACL_SUCCESS) {
        printf("PASS: Got error code as expected (output dtype cast check triggered)\n");
    } else {
        printf("FAIL: Should have returned error for incompatible output dtype (Bug #1 confirmed)\n");
    }

    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
