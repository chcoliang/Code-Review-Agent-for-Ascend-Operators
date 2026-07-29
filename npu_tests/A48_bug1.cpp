#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_batch_matmul.h"
#include <cstdio>
#include <cstdlib>

// A48: self=BF16[2,4,4], mat2=BF16[2,4,4], out=BF16[2,4,4] (expect rejected or accepted)
int main() {
    aclError ret = aclInit(nullptr);
    printf("aclInit: %d\n", ret);
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %d\n", ret);

    int64_t shape[] = {2, 4, 4};
    int64_t strides[] = {16, 4, 1};
    void *devSelf = nullptr, *devMat2 = nullptr, *devOut = nullptr;
    size_t size = 2 * 4 * 4 * 2; // BF16: 2 bytes
    ret = aclrtMalloc(&devSelf, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc self: %d\n", ret);
    ret = aclrtMalloc(&devMat2, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc mat2: %d\n", ret);
    ret = aclrtMalloc(&devOut, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc out: %d\n", ret);

    aclTensor *selfTensor = aclCreateTensor(shape, 3, ACL_BF16, strides, 0,
                                            ACL_FORMAT_ND, shape, 3, devSelf);
    aclTensor *mat2Tensor = aclCreateTensor(shape, 3, ACL_BF16, strides, 0,
                                            ACL_FORMAT_ND, shape, 3, devMat2);
    aclTensor *outTensor = aclCreateTensor(shape, 3, ACL_BF16, strides, 0,
                                           ACL_FORMAT_ND, shape, 3, devOut);

    int8_t cubeMathType = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnBatchMatMulGetWorkspaceSize(selfTensor, mat2Tensor, outTensor,
                                           cubeMathType, &workspaceSize, &executor);
    printf("aclnnBatchMatMulGetWorkspaceSize (BF16): %d, ws=%lu\n", ret, workspaceSize);

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
