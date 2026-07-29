#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_gelu.h"
#include <cassert>
#include <cstdio>
#include <cstdint>

// Bug: 缺少空tensor处理逻辑
// 空tensor应直接返回成功且workspaceSize=0，不进入计算流程
void test_empty_tensor_handling() {
    int64_t shape[] = {2, 0, 3};
    int64_t strides[] = {0, 0, 0};
    aclTensor *self = aclCreateTensor(shape, 3, ACL_FLOAT, strides, 0,
                                       ACL_FORMAT_ND, shape, 3, nullptr);
    aclTensor *out = aclCreateTensor(shape, 3, ACL_FLOAT, strides, 0,
                                      ACL_FORMAT_ND, shape, 3, nullptr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor));

    // 预期: 应返回 ACLNN_SUCCESS 且 workspaceSize == 0, 不进入实际计算
    // 实际: 空tensor被传入l0op算子, 可能导致异常
    printf("A32 Bug1 test_empty_tensor: ret = %d, workspaceSize = %lu\n", (int)ret, (unsigned long)workspaceSize);

    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
}

void test_zero_dim_tensor() {
    int64_t shape[] = {0};
    int64_t strides[] = {1};
    aclTensor *self = aclCreateTensor(shape, 1, ACL_FLOAT16, strides, 0,
                                       ACL_FORMAT_ND, shape, 1, nullptr);
    aclTensor *out = aclCreateTensor(shape, 1, ACL_FLOAT16, strides, 0,
                                      ACL_FORMAT_ND, shape, 1, nullptr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor));

    printf("A32 Bug1 test_zero_dim: ret = %d, workspaceSize = %lu\n", (int)ret, (unsigned long)workspaceSize);

    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
}

int main() {
    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", (int)initRet);
        return -1;
    }

    test_empty_tensor_handling();
    test_zero_dim_tensor();

    aclFinalize();
    return 0;
}
