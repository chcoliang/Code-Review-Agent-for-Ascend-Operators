#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_mul.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Bug: kernel CopyOut 目标类型从 int8_t 改为 uint8_t
// 正确行为: -1*1=-1, -2*1=-2 等 (保持 int8 语义)
// Buggy行为: 结果被按 uint8 写入，读回时解读为正数

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

    // Create INT8 tensors shape=[4]
    int64_t shape[] = {4};
    int64_t strides[] = {1};
    size_t tensorSize = 4 * sizeof(int8_t);

    void *devSelf = nullptr, *devOther = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc self failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOther, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc other failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOut, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc out failed: %d\n", ret); return 1; }

    // Initialize: self=[-1, -2, -3, -4], other=[1, 1, 1, 1]
    int8_t hostSelf[] = {-1, -2, -3, -4};
    int8_t hostOther[] = {1, 1, 1, 1};
    aclrtMemcpy(devSelf, tensorSize, hostSelf, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devOther, tensorSize, hostOther, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclTensor *self = aclCreateTensor(shape, 1, ACL_INT8, strides, 0,
                                       ACL_FORMAT_ND, shape, 1, devSelf);
    aclTensor *other = aclCreateTensor(shape, 1, ACL_INT8, strides, 0,
                                        ACL_FORMAT_ND, shape, 1, devOther);
    aclTensor *out = aclCreateTensor(shape, 1, ACL_INT8, strides, 0,
                                      ACL_FORMAT_ND, shape, 1, devOut);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Test: INT8[4] * INT8[4] -> INT8[4] (CopyOut type mismatch)\n");
    printf("Input self:  [-1, -2, -3, -4]\n");
    printf("Input other: [1, 1, 1, 1]\n");
    printf("Calling aclnnMulGetWorkspaceSize...\n");
    ret = (aclError)aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    printf("Return code: %d\n", (int)ret);

    if (ret == ACL_SUCCESS) {
        void *workspace = nullptr;
        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        }
        ret = (aclError)aclnnMul(workspace, workspaceSize, executor, stream);
        printf("aclnnMul return code: %d\n", (int)ret);
        aclrtSynchronizeStream(stream);

        // Read back results
        int8_t hostOut[4] = {0};
        aclrtMemcpy(hostOut, tensorSize, devOut, tensorSize, ACL_MEMCPY_DEVICE_TO_HOST);
        printf("Results (as int8):  [%d, %d, %d, %d]\n",
               hostOut[0], hostOut[1], hostOut[2], hostOut[3]);
        printf("Expected (correct): [-1, -2, -3, -4]\n");

        // Check if results match expected
        int pass = (hostOut[0] == -1 && hostOut[1] == -2 &&
                    hostOut[2] == -3 && hostOut[3] == -4);
        if (pass) {
            printf("PASS: INT8 CopyOut preserves sign correctly\n");
        } else {
            // If CopyOut uses uint8_t, the bit patterns might be interpreted differently
            uint8_t *uOut = (uint8_t *)hostOut;
            printf("Results (as uint8): [%u, %u, %u, %u]\n",
                   uOut[0], uOut[1], uOut[2], uOut[3]);
            printf("FAIL: Bug confirmed - CopyOut int8→uint8 type mismatch\n");
        }

        if (workspace) aclrtFree(workspace);
    } else {
        printf("FAIL: GetWorkspaceSize returned error %d\n", (int)ret);
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
