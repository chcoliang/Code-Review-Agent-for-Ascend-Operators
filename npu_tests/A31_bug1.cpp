#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_gelu.h"
#include <cassert>
#include <cstdio>
#include <cstdint>

// Bug: 空指针校验失败时返回错误码为 ACLNN_SUCCESS
// CHECK_RET(CheckNotNull(self, out), ACLNN_SUCCESS) 应为 ACLNN_ERR_PARAM_NULLPTR
void test_null_self_returns_wrong_status() {
    aclTensor *self = nullptr;
    int64_t shape[] = {2, 3};
    int64_t strides[] = {3, 1};
    aclTensor *out = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                      ACL_FORMAT_ND, shape, 2, nullptr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor));

    // 预期: 应返回 ACLNN_ERR_PARAM_NULLPTR
    // 实际: 返回 ACLNN_SUCCESS, 然后在self->IsEmpty()处崩溃
    printf("A31 Bug1 test_null_self: ret = %d (expected ACLNN_ERR_PARAM_NULLPTR)\n", (int)ret);

    if (out) aclDestroyTensor(out);
}

void test_null_out_returns_wrong_status() {
    int64_t shape[] = {2, 3};
    int64_t strides[] = {3, 1};
    aclTensor *self = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                       ACL_FORMAT_ND, shape, 2, nullptr);
    aclTensor *out = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor));

    // 预期: 应返回 ACLNN_ERR_PARAM_NULLPTR
    // 实际: 返回 ACLNN_SUCCESS, 然后在访问out成员时崩溃
    printf("A31 Bug1 test_null_out: ret = %d (expected ACLNN_ERR_PARAM_NULLPTR)\n", (int)ret);

    if (self) aclDestroyTensor(self);
}

int main() {
    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", (int)initRet);
        return -1;
    }

    test_null_self_returns_wrong_status();
    test_null_out_returns_wrong_status();

    aclFinalize();
    return 0;
}
