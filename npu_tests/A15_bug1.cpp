#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_mul.h"
#include <cstdio>
#include <cstdlib>

// Bug #1: aclnnMulGetWorkspaceSize Non-Contiguous Path Missing Platform Support Check
// 构造非连续 FP32 tensor，在非 RegBase 平台上直接将非连续 tensor 传入 Mul kernel
// 可能导致计算错误或崩溃
int main() {
    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", initRet);
        return 1;
    }

    int32_t deviceId = 0;
    aclError setRet = aclrtSetDevice(deviceId);
    if (setRet != ACL_SUCCESS) {
        printf("aclrtSetDevice failed: %d\n", setRet);
        aclFinalize();
        return 1;
    }

    // Construct non-contiguous FP32 tensors
    // shape=[4,4] with stride=[8,1] (non-contiguous, gap between rows)
    int64_t shape[] = {4, 4};
    int64_t strides[] = {8, 1};  // non-contiguous stride (contiguous would be {4,1})
    auto self = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);
    auto other = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);

    // out: contiguous FP32 tensor
    int64_t outStrides[] = {4, 1};
    auto out = aclCreateTensor(shape, 2, ACL_FLOAT, outStrides, 0, ACL_FORMAT_ND, shape, 2, nullptr);

    if (self == nullptr || other == nullptr || out == nullptr) {
        printf("Failed to create tensors\n");
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Calling aclnnMulGetWorkspaceSize with non-contiguous FP32 tensors...\n");
    aclError ret = (aclError)aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    printf("Return code: %d\n", (int)ret);
    printf("Workspace size: %lu\n", workspaceSize);

    if (ret == ACL_SUCCESS) {
        printf("INFO: GetWorkspaceSize succeeded. On non-RegBase platforms, the kernel may produce\n");
        printf("      incorrect results if non-contiguous tensors are passed without Contiguous conversion.\n");
        printf("FAIL: Bug #1 confirmed - no platform support check for non-contiguous path\n");
    } else {
        printf("PASS: Got error code (platform correctly rejected non-contiguous input)\n");
    }

    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
