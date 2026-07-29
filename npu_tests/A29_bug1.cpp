#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_gelu.h"
#include <cassert>
#include <cstdio>
#include <cstdint>

// Bug: out参数空指针未校验，会导致空指针解引用崩溃
void test_out_nullptr() {
    int64_t shape[] = {2, 3};
    int64_t strides[] = {3, 1};
    aclTensor *self = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                       ACL_FORMAT_ND, shape, 2, nullptr);

    aclTensor *out = nullptr;  // 空指针
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor));

    // 预期: 应返回 ACLNN_ERR_PARAM_NULLPTR
    // 实际: 由于未校验out，会在后续访问out成员时崩溃
    printf("A29 Bug1 test_out_nullptr: ret = %d (expected ACLNN_ERR_PARAM_NULLPTR)\n", (int)ret);

    if (self) aclDestroyTensor(self);
}

int main() {
    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", (int)initRet);
        return -1;
    }

    test_out_nullptr();

    aclFinalize();
    return 0;
}
