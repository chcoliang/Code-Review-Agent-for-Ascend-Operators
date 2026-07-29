#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_gelu.h"
#include <cstdio>
#include <cstdlib>

// Bug: 删除了 CheckShape 中的 OP_CHECK_MAX_DIM
// 正确行为: 返回错误码 (维度超限, MAX=8)
// Buggy行为: 通过校验，进入后续可能崩溃

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

    // Create 9-dimensional FLOAT32 tensor: shape=[1,1,1,1,1,1,1,1,1] (exceeds MAX_DIM=8)
    int64_t shape[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int64_t strides[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int64_t ndim = 9;
    size_t tensorSize = 1 * sizeof(float); // 1 element total

    void *devSelf = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc self failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOut, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc out failed: %d\n", ret); return 1; }

    aclTensor *self = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                       ACL_FORMAT_ND, shape, ndim, devSelf);
    aclTensor *out = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                      ACL_FORMAT_ND, shape, ndim, devOut);

    if (self == nullptr || out == nullptr) {
        printf("Note: aclCreateTensor may reject 9-dim tensor at API level\n");
        if (self) aclDestroyTensor(self);
        if (out) aclDestroyTensor(out);
        aclrtFree(devSelf);
        aclrtFree(devOut);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 0;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Test: GeLU with 9-dim tensor (exceeds MAX_DIM=8)\n");
    printf("Calling aclnnGeluGetWorkspaceSize...\n");
    ret = (aclError)aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
    printf("Return code: %d\n", (int)ret);
    printf("Workspace size: %lu\n", workspaceSize);

    if (ret != ACL_SUCCESS) {
        printf("PASS: 9-dim tensor correctly rejected with error %d\n", (int)ret);
        printf("Expected behavior: OP_CHECK_MAX_DIM returns dimension error\n");
    } else {
        printf("FAIL: 9-dim tensor was NOT rejected\n");
        printf("Bug confirmed: OP_CHECK_MAX_DIM check was deleted\n");
        printf("Attempting execution (may crash)...\n");

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        }
        ret = (aclError)aclnnGelu(workspace, workspaceSize, executor, stream);
        printf("aclnnGelu return code: %d\n", (int)ret);
        aclrtSynchronizeStream(stream);

        if (workspace) aclrtFree(workspace);
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
