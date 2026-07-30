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

    int64_t mat2Shape[] = {3, 2};
    int64_t outShape[] = {4, 2};
    int64_t mat2Strides[] = {2, 1};
    int64_t outStrides[] = {2, 1};
    void *devMat2 = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devMat2, 3*2*4, ACL_MEM_MALLOC_NORMAL_ONLY);
    ret = aclrtMalloc(&devOut, 4*2*4, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor *selfTensor = nullptr;
    aclTensor *mat2Tensor = aclCreateTensor(mat2Shape, 2, ACL_FLOAT, mat2Strides, 0, ACL_FORMAT_ND, mat2Shape, 2, devMat2);
    aclTensor *outTensor = aclCreateTensor(outShape, 2, ACL_FLOAT, outStrides, 0, ACL_FORMAT_ND, outShape, 2, devOut);
    int8_t cubeMathType = 0;
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    ret = aclnnMatmulGetWorkspaceSize(selfTensor, mat2Tensor, outTensor, cubeMathType, &ws, &exec);
    printf("aclnnMatmulGetWorkspaceSize (self=nullptr): %d\n", ret);
    aclDestroyTensor(mat2Tensor);
    aclDestroyTensor(outTensor);
    aclrtFree(devMat2);
    aclrtFree(devOut);

    aclrtResetDevice(0);
    aclFinalize();
    printf("Done.\n");
    return 0;
}
