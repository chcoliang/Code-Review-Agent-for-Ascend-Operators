#include <cstdio>
#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnnInner_scaled_masked_softmax_v2.h"
int main() {
    aclError ret = aclInit(nullptr);
    ret = aclrtSetDevice(0);
    int64_t xshape[] = {1, 8, 64, 64};
    int64_t mshape[] = {1, 1, 64, 64};
    int64_t xstrides[] = {32768, 4096, 64, 1};
    int64_t mstrides[] = {4096, 4096, 64, 1};
    void *xDev, *mDev, *yDev;
    aclrtMalloc(&xDev, 131072, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&mDev, 4096, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&yDev, 131072, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor *x = aclCreateTensor(xshape, 4, ACL_FLOAT, xstrides, 0, ACL_FORMAT_ND, xshape, 4, xDev);
    aclTensor *mask = aclCreateTensor(mshape, 4, ACL_BOOL, mstrides, 0, ACL_FORMAT_ND, mshape, 4, mDev);
    aclTensor *y = aclCreateTensor(xshape, 4, ACL_FLOAT, xstrides, 0, ACL_FORMAT_ND, xshape, 4, yDev);
    uint64_t ws = 0; aclOpExecutor *exec = nullptr;
    auto st = aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize(x, mask, 1.0, false, y, &ws, &exec);
    printf("aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize: %d, ws=%lu\n", st, ws);
    aclrtFree(xDev); aclrtFree(mDev); aclrtFree(yDev);
    aclrtResetDevice(0); aclFinalize();
    return 0;
}
