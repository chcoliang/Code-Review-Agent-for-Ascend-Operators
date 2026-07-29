#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_mul.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Bug: tiling 层删除了 {DT_FLOAT, DT_FLOAT, DT_FLOAT} 的映射
// 正确行为: FLOAT32 * FLOAT32 → FLOAT32 正常计算
// Buggy行为: Tiling 阶段找不到 float32 映射，返回错误

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

    // Create FLOAT32 tensors shape=[2,2]
    int64_t shape[] = {2, 2};
    int64_t strides[] = {2, 1};
    size_t tensorSize = 4 * sizeof(float); // 4 elements

    void *devSelf = nullptr, *devOther = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc self failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOther, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc other failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOut, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc out failed: %d\n", ret); return 1; }

    // Initialize with some values
    float hostSelf[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hostOther[] = {2.0f, 3.0f, 4.0f, 5.0f};
    aclrtMemcpy(devSelf, tensorSize, hostSelf, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devOther, tensorSize, hostOther, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclTensor *self = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                       ACL_FORMAT_ND, shape, 2, devSelf);
    aclTensor *other = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                        ACL_FORMAT_ND, shape, 2, devOther);
    aclTensor *out = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                      ACL_FORMAT_ND, shape, 2, devOut);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Test: FLOAT32[2,2] * FLOAT32[2,2] -> FLOAT32[2,2]\n");
    printf("Calling aclnnMulGetWorkspaceSize...\n");
    ret = (aclError)aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    printf("Return code: %d\n", (int)ret);
    printf("Workspace size: %lu\n", workspaceSize);

    if (ret == ACL_SUCCESS) {
        void *workspace = nullptr;
        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        }
        ret = (aclError)aclnnMul(workspace, workspaceSize, executor, stream);
        printf("aclnnMul return code: %d\n", (int)ret);
        aclrtSynchronizeStream(stream);

        // Read back results
        float hostOut[4] = {0};
        aclrtMemcpy(hostOut, tensorSize, devOut, tensorSize, ACL_MEMCPY_DEVICE_TO_HOST);
        printf("Results: [%.1f, %.1f, %.1f, %.1f]\n",
               hostOut[0], hostOut[1], hostOut[2], hostOut[3]);
        printf("Expected: [2.0, 6.0, 12.0, 20.0]\n");

        if (workspace) aclrtFree(workspace);
        printf("PASS: FLOAT32 mul completed successfully\n");
    } else {
        printf("FAIL: GetWorkspaceSize returned error %d\n", (int)ret);
        printf("Bug confirmed: DTYPE_MAP missing DT_FLOAT mapping causes tiling failure\n");
    }

    // Cleanup
    aclDestroyTensor(out);
    aclDestroyTensor(other);
    aclDestroyTensor(self);
    aclrtFree(devOut);
    aclrtFree(devOther);
    aclrtFree(devSelf);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
