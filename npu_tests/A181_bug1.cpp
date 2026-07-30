#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnn/opdev/op_errno.h"

#include <cstdio>
#include <cstdint>
#include <csignal>
#include <cstdlib>

void segfault_handler(int sig) {
    printf("BUG CONFIRMED: Caught SEGFAULT (signal %d)\n", sig);
    printf("  Root cause: CheckMulNotNull does not validate 'out' parameter.\n");
    printf("  Expected behavior: aclnnMulGetWorkspaceSize should return ACLNN_ERR_PARAM_NULLPTR (161001)\n");
    exit(1);
}

int main() {
    signal(SIGSEGV, segfault_handler);

    // 初始化ACL
    aclError ret = aclInit(nullptr);
    if (ret != 0) {
        printf("aclInit failed: %d\n", ret);
        return 1;
    }
    ret = aclrtSetDevice(0);
    if (ret != 0) {
        printf("aclrtSetDevice failed: %d\n", ret);
        return 1;
    }

    // 创建有效的 self tensor: shape=[2,3], dtype=FLOAT
    int64_t selfShape[] = {2, 3};
    int64_t selfStrides[] = {3, 1};
    void* selfDevPtr = nullptr;
    aclrtMalloc(&selfDevPtr, 2 * 3 * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor* self = aclCreateTensor(selfShape, 2, ACL_FLOAT, selfStrides, 0,
                                       ACL_FORMAT_ND, selfShape, 2, selfDevPtr);

    // 创建有效的 other tensor: shape=[2,3], dtype=FLOAT
    void* otherDevPtr = nullptr;
    aclrtMalloc(&otherDevPtr, 2 * 3 * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor* other = aclCreateTensor(selfShape, 2, ACL_FLOAT, selfStrides, 0,
                                        ACL_FORMAT_ND, selfShape, 2, otherDevPtr);

    // out 设为 nullptr —— 触发 bug
    aclTensor* out = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    printf("Calling aclnnMulGetWorkspaceSize with out=nullptr...\n");
    printf("Expected: return ACLNN_ERR_PARAM_NULLPTR without crash\n");
    printf("Actual: ");
    fflush(stdout);

    // 此调用应触发 SEGFAULT，因为 CheckMulNotNull 未检查 out
    aclnnStatus status = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

    // 如果没有崩溃（不太可能到达这里）
    printf("returned status = %d\n", (int)status);
    if (status == 0) {
        printf("BUG: Should have returned error for nullptr out, but got SUCCESS\n");
    } else {
        printf("Returned error code %d (may have been fixed)\n", (int)status);
    }

    // 清理
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDevPtr);
    aclrtFree(otherDevPtr);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
