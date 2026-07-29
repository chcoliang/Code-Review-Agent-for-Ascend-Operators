#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_mul.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Bug: InferTensorScalarDtype 中删除了 keepB16 精度检查
// FP16 tensor × 大 scalar (65536.0 超出 FP16 max 65504) 不会提升到 FP32
// 正确行为: 应检测到 scalar 超出 FP16 范围，提升到 FP32 计算
// Buggy行为: 保持 FP16 计算，结果溢出为 INF

int main() {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", ret);
        return 1;
    }

    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSetDevice failed: %d\n", ret);
        aclFinalize();
        return 1;
    }

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtCreateStream failed: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    // Create FP16 self tensor shape=[4]
    int64_t shape[] = {4};
    int64_t strides[] = {1};
    size_t selfSize = 4 * 2; // 4 elements * 2 bytes (FP16)

    void *devSelf = nullptr;
    ret = aclrtMalloc(&devSelf, selfSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        printf("aclrtMalloc devSelf failed: %d\n", ret);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    aclTensor *self = aclCreateTensor(shape, 1, ACL_FLOAT16, strides, 0,
                                       ACL_FORMAT_ND, shape, 1, devSelf);
    if (self == nullptr) {
        printf("Failed to create self tensor\n");
        aclrtFree(devSelf);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    // Create output tensor (FP16)
    void *devOut = nullptr;
    ret = aclrtMalloc(&devOut, selfSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        printf("aclrtMalloc devOut failed: %d\n", ret);
        aclDestroyTensor(self);
        aclrtFree(devSelf);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    aclTensor *out = aclCreateTensor(shape, 1, ACL_FLOAT16, strides, 0,
                                      ACL_FORMAT_ND, shape, 1, devOut);

    // Create scalar = 65536.0 (exceeds FP16 max 65504)
    float scalarValue = 65536.0f;
    aclScalar *scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Test: FP16 tensor * scalar(65536.0) - scalar exceeds FP16 max\n");
    printf("Calling aclnnMulsGetWorkspaceSize...\n");
    ret = (aclError)aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
    printf("Return code: %d\n", (int)ret);
    printf("Workspace size: %lu\n", workspaceSize);

    if (ret == ACL_SUCCESS) {
        printf("GetWorkspaceSize succeeded.\n");
        printf("Correct behavior: scalar should be promoted to FP32 to avoid overflow\n");
        printf("Buggy behavior: stays FP16, result will overflow to INF\n");

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        }
        ret = (aclError)aclnnMuls(workspace, workspaceSize, executor, stream);
        printf("aclnnMuls return code: %d\n", (int)ret);
        aclrtSynchronizeStream(stream);

        if (workspace) aclrtFree(workspace);
    } else {
        printf("FAIL at GetWorkspaceSize stage: %d\n", (int)ret);
    }

    // Cleanup
    aclDestroyScalar(scalar);
    aclDestroyTensor(out);
    aclDestroyTensor(self);
    aclrtFree(devOut);
    aclrtFree(devSelf);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
