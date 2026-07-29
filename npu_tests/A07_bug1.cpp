#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnn/opdev/op_errno.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>

int main() {
    // 初始化 ACL
    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", (int)aclRet);
        return 1;
    }

    aclRet = aclrtSetDevice(0);
    if (aclRet != ACL_SUCCESS) {
        printf("aclrtSetDevice failed: %d\n", (int)aclRet);
        return 1;
    }

    // 传入 nullptr 作为 self 参数
    const aclTensor *self = nullptr;
    const aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Calling aclnnMulGetWorkspaceSize with self=nullptr, other=nullptr, out=nullptr\n");
    printf("Expected behavior: should return ACLNN_ERR_PARAM_NULLPTR (161001)\n");
    printf("Actual behavior due to bug: returns ACLNN_SUCCESS (0) then crashes (SEGFAULT)\n\n");

    // 此调用应该返回 ACLNN_ERR_PARAM_NULLPTR，
    // 但由于 bug，CheckMulParams 返回 ACLNN_SUCCESS，
    // 随后代码继续执行并解引用 nullptr，导致 SEGFAULT
    aclnnStatus status = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

    // 如果没有崩溃（理论上不应该到达这里）
    printf("Return status: %d\n", (int)status);
    if (status == 0) {
        printf("BUG CONFIRMED: returned ACLNN_SUCCESS (0) for nullptr input!\n");
        printf("Correct behavior: should have returned ACLNN_ERR_PARAM_NULLPTR (161001)\n");
    }

    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
