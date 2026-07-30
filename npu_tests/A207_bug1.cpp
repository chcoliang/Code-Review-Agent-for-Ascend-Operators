#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_gelu.h"
#include <cstdio>
#include <cstdlib>

// Bug #1: BF16 数据类型永远无法通过校验（逻辑矛盾）
// 在 910B 上 BF16 应该被支持，但 DTYPE_SUPPORT_LIST 不含 BF16，导致被拒绝
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

    // self: shape=[2,3], dtype=BF16
    int64_t shape[] = {2, 3};
    int64_t strides[] = {3, 1};
    auto self = aclCreateTensor(shape, 2, ACL_BF16, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);

    // out: shape=[2,3], dtype=BF16
    auto out = aclCreateTensor(shape, 2, ACL_BF16, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);

    if (self == nullptr || out == nullptr) {
        printf("Failed to create BF16 tensors\n");
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Calling aclnnGeluGetWorkspaceSize with BF16 tensors on 910B...\n");
    aclError ret = (aclError)aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
    printf("Return code: %d\n", (int)ret);

    if (ret == ACL_SUCCESS) {
        printf("PASS: BF16 accepted on 910B as expected\n");
    } else {
        printf("FAIL: BF16 rejected on 910B (Bug #1 confirmed - dtype support list missing BF16)\n");
    }

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
