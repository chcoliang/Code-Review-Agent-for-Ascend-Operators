#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_gelu.h"
#include <cstdio>
#include <cstdlib>

int main() {
    aclError ret = aclInit(nullptr);
    printf("aclInit: %d\n", ret);
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %d\n", ret);

    // GeLU with large tensor [1024,1024] to test tiling
    int64_t shape[] = {1024, 1024};
    int64_t strides[] = {1024, 1};
    void *devSelf = nullptr, *devOut = nullptr;
    size_t size = 1024 * 1024 * 4; // FP32
    ret = aclrtMalloc(&devSelf, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc self: %d\n", ret);
    ret = aclrtMalloc(&devOut, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc out: %d\n", ret);

    aclTensor *selfTensor = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                            ACL_FORMAT_ND, shape, 2, devSelf);
    aclTensor *outTensor = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                           ACL_FORMAT_ND, shape, 2, devOut);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGeluGetWorkspaceSize(selfTensor, outTensor, &workspaceSize, &executor);
    printf("aclnnGeluGetWorkspaceSize (large [1024,1024]): %d, workspaceSize=%lu\n", ret, workspaceSize);

    aclDestroyTensor(selfTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(devSelf);
    aclrtFree(devOut);
    aclrtResetDevice(0);
    aclFinalize();
    printf("Done.\n");
    return 0;
}
