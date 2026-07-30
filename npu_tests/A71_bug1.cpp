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
    int64_t selfStrides[] = {3, 1};
    int64_t mat2Strides[] = {2, 1};
    void *devSelf = nullptr, *devMat2 = nullptr;
    ret = aclrtMalloc(&devSelf, 4*3*4, ACL_MEM_MALLOC_NORMAL_ONLY);
    ret = aclrtMalloc(&devMat2, 3*2*4, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor *selfTensor = aclCreateTensor(selfShape, 2, ACL_FLOAT, selfStrides, 0, ACL_FORMAT_ND, selfShape, 2, devSelf);
    aclTensor *mat2Tensor = aclCreateTensor(mat2Shape, 2, ACL_FLOAT, mat2Strides, 0, ACL_FORMAT_ND, mat2Shape, 2, devMat2);
    aclTensor *outTensor = nullptr;
    int8_t cubeMathType = 0;
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    ret = aclnnMatmulGetWorkspaceSize(selfTensor, mat2Tensor, outTensor, cubeMathType, &ws, &exec);
    printf("aclnnMatmulGetWorkspaceSize (out=nullptr): %d\n", ret);
    aclDestroyTensor(selfTensor);
    aclDestroyTensor(mat2Tensor);
    aclrtFree(devSelf);
    aclrtFree(devMat2);

    aclrtResetDevice(0);
    aclFinalize();
    printf("Done.\n");
    return 0;
}
