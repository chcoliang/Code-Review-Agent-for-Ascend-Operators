#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_softmax.h"
#include <cstdio>
#include <cstdlib>

int main() {
    aclError ret = aclInit(nullptr);
    printf("aclInit: %d\n", ret);
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %d\n", ret);

    // Softmax: transposed tensor (strides=[1,4] instead of [4,1]) - non-contiguous
    int64_t shape[] = {2, 4};
    int64_t strides[] = {1, 4}; // transposed strides
    void *devSelf = nullptr, *devOut = nullptr;
    size_t size = 2 * 4 * 4; // FP32
    ret = aclrtMalloc(&devSelf, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc self: %d\n", ret);
    ret = aclrtMalloc(&devOut, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc out: %d\n", ret);

    aclTensor *selfTensor = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                            ACL_FORMAT_ND, shape, 2, devSelf);
    int64_t outStrides[] = {4, 1};
    aclTensor *outTensor = aclCreateTensor(shape, 2, ACL_FLOAT, outStrides, 0,
                                           ACL_FORMAT_ND, shape, 2, devOut);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int64_t dim = 1;
    ret = aclnnSoftmaxGetWorkspaceSize(selfTensor, dim, outTensor, &workspaceSize, &executor);
    printf("aclnnSoftmaxGetWorkspaceSize (non-contiguous): %d, workspaceSize=%lu\n", ret, workspaceSize);

    aclDestroyTensor(selfTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(devSelf);
    aclrtFree(devOut);
    aclrtResetDevice(0);
    aclFinalize();
    printf("Done.\n");
    return 0;
}
