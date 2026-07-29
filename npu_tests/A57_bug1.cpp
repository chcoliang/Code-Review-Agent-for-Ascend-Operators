#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_batch_matmul.h"
#include <cstdio>
#include <cstdlib>

// A57: self=FP16[2,4,3], mat2=FP16[2,5,4] (K mismatch: 3!=5)
int main() {
    aclError ret = aclInit(nullptr);
    printf("aclInit: %d\n", ret);
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %d\n", ret);

    int64_t shapeSelf[] = {2, 4, 3};
    int64_t stridesSelf[] = {12, 3, 1};
    int64_t shapeMat2[] = {2, 5, 4};
    int64_t stridesMat2[] = {20, 4, 1};
    int64_t shapeOut[] = {2, 4, 4};
    int64_t stridesOut[] = {16, 4, 1};

    void *devSelf = nullptr, *devMat2 = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, 2*4*3*2, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc self: %d\n", ret);
    ret = aclrtMalloc(&devMat2, 2*5*4*2, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc mat2: %d\n", ret);
    ret = aclrtMalloc(&devOut, 2*4*4*2, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc out: %d\n", ret);

    aclTensor *selfTensor = aclCreateTensor(shapeSelf, 3, ACL_FLOAT16, stridesSelf, 0,
                                            ACL_FORMAT_ND, shapeSelf, 3, devSelf);
    aclTensor *mat2Tensor = aclCreateTensor(shapeMat2, 3, ACL_FLOAT16, stridesMat2, 0,
                                            ACL_FORMAT_ND, shapeMat2, 3, devMat2);
    aclTensor *outTensor = aclCreateTensor(shapeOut, 3, ACL_FLOAT16, stridesOut, 0,
                                           ACL_FORMAT_ND, shapeOut, 3, devOut);

    int8_t cubeMathType = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnBatchMatMulGetWorkspaceSize(selfTensor, mat2Tensor, outTensor,
                                           cubeMathType, &workspaceSize, &executor);
    printf("aclnnBatchMatMulGetWorkspaceSize (K mismatch 3!=5): %d\n", ret);
    printf("Expected: shape mismatch error\n");

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
