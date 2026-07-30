#include <cstdio>
#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_apply_adam_w.h"
int main() {
    aclInit(nullptr); aclrtSetDevice(0);
    int64_t shape[] = {4, 4}; int64_t strides[] = {4, 1};
    int64_t sshape[] = {1}; int64_t sstrides[] = {1};
    void *dev[12];
    for (int i = 0; i < 12; i++) aclrtMalloc(&dev[i], 64, ACL_MEM_MALLOC_NORMAL_ONLY);
    // varRef, mRef, vRef (mutable)
    aclTensor *varRef = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, dev[0]);
    aclTensor *mRef = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, dev[1]);
    aclTensor *vRef = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, dev[2]);
    // scalars as 1-element tensors
    aclTensor *beta1Power = aclCreateTensor(sshape, 1, ACL_FLOAT, sstrides, 0, ACL_FORMAT_ND, sshape, 1, dev[3]);
    aclTensor *beta2Power = aclCreateTensor(sshape, 1, ACL_FLOAT, sstrides, 0, ACL_FORMAT_ND, sshape, 1, dev[4]);
    aclTensor *lr = aclCreateTensor(sshape, 1, ACL_FLOAT, sstrides, 0, ACL_FORMAT_ND, sshape, 1, dev[5]);
    aclTensor *weightDecay = aclCreateTensor(sshape, 1, ACL_FLOAT, sstrides, 0, ACL_FORMAT_ND, sshape, 1, dev[6]);
    aclTensor *beta1 = aclCreateTensor(sshape, 1, ACL_FLOAT, sstrides, 0, ACL_FORMAT_ND, sshape, 1, dev[7]);
    aclTensor *beta2 = aclCreateTensor(sshape, 1, ACL_FLOAT, sstrides, 0, ACL_FORMAT_ND, sshape, 1, dev[8]);
    aclTensor *eps = aclCreateTensor(sshape, 1, ACL_FLOAT, sstrides, 0, ACL_FORMAT_ND, sshape, 1, dev[9]);
    aclTensor *grad = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, dev[10]);
    uint64_t ws = 0; aclOpExecutor *exec = nullptr;
    auto st = aclnnApplyAdamWGetWorkspaceSize(varRef, mRef, vRef, beta1Power, beta2Power, lr, weightDecay, beta1, beta2, eps, grad, nullptr, false, false, &ws, &exec);
    printf("aclnnApplyAdamWGetWorkspaceSize: %d, ws=%lu\n", st, ws);
    for (int i = 0; i < 12; i++) aclrtFree(dev[i]);
    aclrtResetDevice(0); aclFinalize();
    return 0;
}
