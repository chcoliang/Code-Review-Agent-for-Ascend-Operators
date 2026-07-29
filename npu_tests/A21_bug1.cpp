#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_softmax.h"
#include <cstdio>
#include <cstdlib>

// Bug: 删除了 aclnnSoftmaxGetWorkspaceSize 中的空tensor提前返回
// 正确行为: 直接返回 SUCCESS, workspaceSize=0
// Buggy行为: 继续执行计算图构建，可能超时或报错

int main() {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", ret);
        return 1;
    }

    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSetDevice failed: %d\n", ret);
        aclFinalize();
        return 1;
    }

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtCreateStream failed: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    // Create empty FLOAT32 tensor: shape=[0,4]
    int64_t shape[] = {0, 4};
    int64_t strides[] = {4, 1};

    // For empty tensor, we still need a valid device pointer (can be minimal)
    void *devSelf = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, 4, ACL_MEM_MALLOC_NORMAL_ONLY); // minimal alloc
    if (ret != ACL_SUCCESS) { printf("malloc self failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOut, 4, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc out failed: %d\n", ret); return 1; }

    aclTensor *self = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                       ACL_FORMAT_ND, shape, 2, devSelf);
    aclTensor *out = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                      ACL_FORMAT_ND, shape, 2, devOut);

    if (self == nullptr || out == nullptr) {
        printf("Failed to create tensors\n");
        if (self) aclDestroyTensor(self);
        if (out) aclDestroyTensor(out);
        aclrtFree(devSelf);
        aclrtFree(devOut);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int64_t dim = 1;

    printf("Test: Softmax with empty tensor shape=[0,4], dim=1\n");
    printf("Calling aclnnSoftmaxGetWorkspaceSize...\n");
    ret = (aclError)aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);
    printf("Return code: %d\n", (int)ret);
    printf("Workspace size: %lu\n", workspaceSize);

    if (ret == ACL_SUCCESS && workspaceSize == 0) {
        printf("PASS: Empty tensor handled correctly (early return, workspaceSize=0)\n");
    } else if (ret == ACL_SUCCESS && workspaceSize > 0) {
        printf("FAIL: Empty tensor should not require workspace\n");
        printf("Bug confirmed: missing empty tensor early return\n");
    } else {
        printf("FAIL: Unexpected error %d for empty tensor\n", (int)ret);
        printf("Bug confirmed: missing empty tensor check causes computation failure\n");
    }

    // Cleanup
    aclDestroyTensor(out);
    aclDestroyTensor(self);
    aclrtFree(devOut);
    aclrtFree(devSelf);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
