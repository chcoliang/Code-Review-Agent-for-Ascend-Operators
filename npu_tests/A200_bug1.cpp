#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_softmax.h"
#include <cstdio>
#include <cstdlib>

// Bug #1: out 参数未做空指针校验，传入 nullptr 应返回错误码，实际会 segfault
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

    // Create a valid self tensor: shape=[2,3], dtype=float32
    int64_t shape[] = {2, 3};
    int64_t strides[] = {3, 1};
    auto self = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);
    if (self == nullptr) {
        printf("Failed to create self tensor\n");
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    // Bug trigger: pass nullptr as out tensor
    printf("Calling aclnnSoftmaxGetWorkspaceSize with out=nullptr...\n");
    aclError ret = (aclError)aclnnSoftmaxGetWorkspaceSize(self, 0, nullptr, &workspaceSize, &executor);
    printf("Return code: %d\n", (int)ret);

    if (ret != ACL_SUCCESS) {
        printf("PASS: Got error code as expected (nullptr out rejected)\n");
    } else {
        printf("FAIL: Should have returned error for nullptr out\n");
    }

    aclDestroyTensor(self);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
