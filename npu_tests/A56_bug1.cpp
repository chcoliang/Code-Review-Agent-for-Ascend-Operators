#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_batch_matmul.h"
#include <cstdio>
#include <cstdlib>

// A56: self=FP16[4,4] (2D instead of 3D, expect dimension error)
int main() {
    aclError ret = aclInit(nullptr);
    printf("aclInit: %d\n", ret);
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %d\n", ret);

    int64_t shape2D[] = {4, 4};
    int64_t strides2D[] = {4, 1};
    int64_t shape3D[] = {2, 4, 4};
    int64_t strides3D[] = {16, 4, 1};
    void *devSelf = nullptr, *devMat2 = nullptr, *devOut = nullptr;
    size_t size2D = 4 * 4 * 2;
    size_t size3D = 2 * 4 * 4 * 2;
    ret = aclrtMalloc(&devSelf, size2D, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc self: %d\n", ret);
    ret = aclrtMalloc(&devMat2, size3D, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc mat2: %d\n", ret);
    ret = aclrtMalloc(&devOut, size3D, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc out: %d\n", ret);

    aclTensor *selfTensor = aclCreateTensor(shape2D, 2, ACL_FLOAT16, strides2D, 0,
                                            ACL_FORMAT_ND, shape2D, 2, devSelf);
    aclTensor *mat2Tensor = aclCreateTensor(shape3D, 3, ACL_FLOAT16, strides3D, 0,
                                            ACL_FORMAT_ND, shape3D, 3, devMat2);
    aclTensor *outTensor = aclCreateTensor(shape3D, 3, ACL_FLOAT16, strides3D, 0,
                                           ACL_FORMAT_ND, shape3D, 3, devOut);

    int8_t cubeMathType = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnBatchMatMulGetWorkspaceSize(selfTensor, mat2Tensor, outTensor,
                                           cubeMathType, &workspaceSize, &executor);
    printf("aclnnBatchMatMulGetWorkspaceSize (2D self): %d\n", ret);
    printf("Expected: dimension error\n");

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
