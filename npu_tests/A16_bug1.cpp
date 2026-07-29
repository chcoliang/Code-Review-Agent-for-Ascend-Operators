#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_mul.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Bug: kernel 中删除了 AndFF 截断
// 正确行为: 100*2=200 → &0xFF → 200(uint8) → -56(int8)
// Buggy行为: 100*2=200 不截断，cast 到 int8 时结果不同

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
    size_t tensorSize = 4 * sizeof(int8_t); // 4 bytes

    void *devSelf = nullptr, *devOther = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc self failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOther, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc other failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOut, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc out failed: %d\n", ret); return 1; }

    // Initialize: self=[100, -100, 127, -128], other=[2, 2, 2, 2]
    int8_t hostSelf[] = {100, -100, 127, -128};
    int8_t hostOther[] = {2, 2, 2, 2};
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

    printf("Test: INT8[4] * INT8[4] -> INT8[4] (overflow truncation)\n");
    printf("Input self:  [100, -100, 127, -128]\n");
    printf("Input other: [2, 2, 2, 2]\n");
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
        printf("Results:  [%d, %d, %d, %d]\n",
               hostOut[0], hostOut[1], hostOut[2], hostOut[3]);
        // With correct &0xFF truncation:
        // 100*2=200 → &0xFF=200 → (int8_t)200 = -56
        // -100*2=-200 → &0xFF=56 → (int8_t)56 = 56
        // 127*2=254 → &0xFF=254 → (int8_t)254 = -2
        // -128*2=-256 → &0xFF=0 → (int8_t)0 = 0
        printf("Expected (with &0xFF): [-56, 56, -2, 0]\n");
        printf("Buggy (without &0xFF): results differ due to missing truncation\n");

        int pass = (hostOut[0] == -56 && hostOut[1] == 56 &&
                    hostOut[2] == -2 && hostOut[3] == 0);
        printf("%s: INT8 overflow truncation %s\n", pass ? "PASS" : "FAIL",
               pass ? "works correctly" : "missing - bug confirmed");

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
