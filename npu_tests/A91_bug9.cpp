#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_swish.h"
#include <cstdio>
#include <cstdlib>

int main() {
    aclError ret = aclInit(nullptr);
    printf("aclInit: %d\n", ret);
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %d\n", ret);

    int64_t shape[] = {4, 4};
    int64_t strides[] = {1, 4};
    void *devSelf = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, 4*4*4, ACL_MEM_MALLOC_NORMAL_ONLY);
    ret = aclrtMalloc(&devOut, 4*4*4, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor *selfTensor = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, devSelf);
    int64_t outStrides[] = {4, 1};
    aclTensor *outTensor = aclCreateTensor(shape, 2, ACL_FLOAT, outStrides, 0, ACL_FORMAT_ND, shape, 2, devOut);
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    ret = aclnnSwishGetWorkspaceSize(selfTensor, outTensor, &ws, &exec);
    printf("aclnnSwishGetWorkspaceSize (non-contiguous): %d, ws=%lu\n", ret, ws);
    aclDestroyTensor(selfTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(devSelf);
    aclrtFree(devOut);

    aclrtResetDevice(0);
    aclFinalize();
    printf("Done.\n");
    return 0;
}
