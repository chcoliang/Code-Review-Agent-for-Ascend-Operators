#include <cstdio>
#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_convolution.h"
int main() {
    aclError ret = aclInit(nullptr);
    ret = aclrtSetDevice(0);
    int64_t shape[] = {1, 3, 4, 4};
    int64_t wshape[] = {6, 3, 3, 3};
    int64_t oshape[] = {1, 6, 2, 2};
    int64_t strides[] = {48, 16, 4, 1};
    int64_t wstrides[] = {27, 9, 3, 1};
    int64_t ostrides[] = {24, 4, 2, 1};
    void *selfDev, *wDev, *oDev;
    aclrtMalloc(&selfDev, 192, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&wDev, 648, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&oDev, 96, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor *input = aclCreateTensor(shape, 4, ACL_FLOAT, strides, 0, ACL_FORMAT_NCHW, shape, 4, selfDev);
    aclTensor *weight = aclCreateTensor(wshape, 4, ACL_FLOAT, wstrides, 0, ACL_FORMAT_NCHW, wshape, 4, wDev);
    aclTensor *out = aclCreateTensor(oshape, 4, ACL_FLOAT, ostrides, 0, ACL_FORMAT_NCHW, oshape, 4, oDev);
    int64_t strideArr[] = {1, 1};
    int64_t padArr[] = {0, 0};
    int64_t dilArr[] = {1, 1};
    aclIntArray *stride_a = aclCreateIntArray(strideArr, 2);
    aclIntArray *pad_a = aclCreateIntArray(padArr, 2);
    aclIntArray *dil_a = aclCreateIntArray(dilArr, 2);
    uint64_t ws = 0; aclOpExecutor *exec = nullptr;
    auto st = aclnnConvolutionGetWorkspaceSize(input, weight, nullptr, stride_a, pad_a, dil_a, false, pad_a, 1, out, 0, &ws, &exec);
    printf("aclnnConvolutionGetWorkspaceSize: %d, ws=%lu\n", st, ws);
    aclrtFree(selfDev); aclrtFree(wDev); aclrtFree(oDev);
    aclrtResetDevice(0); aclFinalize();
    return 0;
}
