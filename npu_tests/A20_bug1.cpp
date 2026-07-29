#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_softmax.h"
#include <cstdio>
#include <cstdlib>

// Bug #1: 输出 tensor shape 未校验与输入一致
// self shape=[2,3,4], out shape=[2,3] (shape mismatch)
// 预期返回 ACLNN_ERR_PARAM_INVALID，实际进入计算流程导致未定义行为
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

    // self: shape=[2,3,4], dtype=float32
    int64_t selfShape[] = {2, 3, 4};
    int64_t selfStrides[] = {12, 4, 1};
    auto self = aclCreateTensor(selfShape, 3, ACL_FLOAT, selfStrides, 0, ACL_FORMAT_ND, selfShape, 3, nullptr);

    // out: shape=[2,3] (mismatched shape)
    int64_t outShape[] = {2, 3};
    int64_t outStrides[] = {3, 1};
    auto out = aclCreateTensor(outShape, 2, ACL_FLOAT, outStrides, 0, ACL_FORMAT_ND, outShape, 2, nullptr);

    if (self == nullptr || out == nullptr) {
        printf("Failed to create tensors\n");
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Calling aclnnSoftmaxGetWorkspaceSize with mismatched out shape...\n");
    aclError ret = (aclError)aclnnSoftmaxGetWorkspaceSize(self, 0, out, &workspaceSize, &executor);
    printf("Return code: %d\n", (int)ret);

    if (ret != ACL_SUCCESS) {
        printf("PASS: Got error code as expected (shape mismatch rejected)\n");
    } else {
        printf("FAIL: Should have returned error for shape mismatch\n");
    }

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
