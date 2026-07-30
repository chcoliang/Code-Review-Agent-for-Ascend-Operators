#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_leaky_relu.h"
#include <cstdio>
#include <cstdlib>

int main() {
    aclError ret = aclInit(nullptr);
    printf("aclInit: %d\n", ret);
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %d\n", ret);

    int64_t selfShape[] = {4, 4};
    int64_t outShape[] = {4, 3};
    int64_t selfStrides[] = {4, 1};
    int64_t outStrides[] = {3, 1};
    void *devSelf = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, 4*4*4, ACL_MEM_MALLOC_NORMAL_ONLY);
    ret = aclrtMalloc(&devOut, 4*3*4, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor *selfTensor = aclCreateTensor(selfShape, 2, ACL_FLOAT, selfStrides, 0, ACL_FORMAT_ND, selfShape, 2, devSelf);
    aclTensor *outTensor = aclCreateTensor(outShape, 2, ACL_FLOAT, outStrides, 0, ACL_FORMAT_ND, outShape, 2, devOut);
    float nsVal = 0.01f;
    aclScalar *negativeSlope = aclCreateScalar(&nsVal, ACL_FLOAT);
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    ret = aclnnLeakyReluGetWorkspaceSize(selfTensor, negativeSlope, outTensor, &ws, &exec);
    printf("aclnnLeakyReluGetWorkspaceSize (shape mismatch): %d, ws=%lu\n", ret, ws);
    aclDestroyTensor(selfTensor);
    aclDestroyTensor(outTensor);
    aclDestroyScalar(negativeSlope);
    aclrtFree(devSelf);
    aclrtFree(devOut);

    aclrtResetDevice(0);
    aclFinalize();
    printf("Done.\n");
    return 0;
}
