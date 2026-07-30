#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_matmul.h"
#include <cstdio>
#include <cstdlib>

int main() {
    aclError ret = aclInit(nullptr);
    printf("aclInit: %d\n", ret);
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %d\n", ret);

    int64_t selfShape[] = {4, 3};
    int64_t mat2Shape[] = {3, 2};
    int64_t outShape[] = {4, 2};
    int64_t selfStrides[] = {3, 1};
    int64_t mat2Strides[] = {2, 1};
    int64_t outStrides[] = {2, 1};
    void *devSelf = nullptr, *devMat2 = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, 4*3*2, ACL_MEM_MALLOC_NORMAL_ONLY);
    ret = aclrtMalloc(&devMat2, 3*2*2, ACL_MEM_MALLOC_NORMAL_ONLY);
    ret = aclrtMalloc(&devOut, 4*2*2, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor *selfTensor = aclCreateTensor(selfShape, 2, ACL_BF16, selfStrides, 0, ACL_FORMAT_ND, selfShape, 2, devSelf);
    aclTensor *mat2Tensor = aclCreateTensor(mat2Shape, 2, ACL_BF16, mat2Strides, 0, ACL_FORMAT_ND, mat2Shape, 2, devMat2);
    aclTensor *outTensor = aclCreateTensor(outShape, 2, ACL_BF16, outStrides, 0, ACL_FORMAT_ND, outShape, 2, devOut);
    int8_t cubeMathType = 0;
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    ret = aclnnMatmulGetWorkspaceSize(selfTensor, mat2Tensor, outTensor, cubeMathType, &ws, &exec);
    printf("aclnnMatmulGetWorkspaceSize (BF16): %d, ws=%lu\n", ret, ws);
    aclDestroyTensor(selfTensor);
    aclDestroyTensor(mat2Tensor);
    aclDestroyTensor(outTensor);
    aclrtFree(devSelf);
    aclrtFree(devMat2);
    aclrtFree(devOut);

    aclrtResetDevice(0);
    aclFinalize();
    printf("Done.\n");
    return 0;
}
