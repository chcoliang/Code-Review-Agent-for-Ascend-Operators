#include <cstdio>
#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_mul.h"
int main() {
    aclInit(nullptr); aclrtSetDevice(0);
    int64_t shape[]={4}; int64_t strides[]={1};
    void *dev; aclrtMalloc(&dev, 16, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor *selfRef = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 1, dev);
    uint64_t ws=0; aclOpExecutor *exec=nullptr;
    // InplaceMul with other=nullptr (bug: should be rejected but isn't in buggy version)
    auto st = aclnnInplaceMulGetWorkspaceSize(selfRef, nullptr, &ws, &exec);
    printf("aclnnInplaceMulGetWorkspaceSize(other=nullptr): %d\n", st);
    printf("Expected: ACLNN_ERR_PARAM_NULLPTR (161001)\n");
    aclrtFree(dev); aclrtResetDevice(0); aclFinalize();
    return 0;
}
