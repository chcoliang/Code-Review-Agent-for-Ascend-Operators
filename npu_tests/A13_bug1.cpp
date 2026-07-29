#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_mul.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Bug: opFile 名称错误 (mul_opt 不存在)
// 正确行为: 正常计算
// Buggy行为: kernel 文件找不到，执行失败

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

    // Create FLOAT32 tensors shape=[4]
    int64_t shape[] = {4};
    int64_t strides[] = {1};
    size_t tensorSize = 4 * sizeof(float);

    void *devSelf = nullptr, *devOther = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc self failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOther, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc other failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOut, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc out failed: %d\n", ret); return 1; }

    // Initialize
    float hostSelf[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hostOther[] = {5.0f, 6.0f, 7.0f, 8.0f};
    aclrtMemcpy(devSelf, tensorSize, hostSelf, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devOther, tensorSize, hostOther, tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclTensor *self = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0,
                                       ACL_FORMAT_ND, shape, 1, devSelf);
    aclTensor *other = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0,
                                        ACL_FORMAT_ND, shape, 1, devOther);
    aclTensor *out = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0,
                                      ACL_FORMAT_ND, shape, 1, devOut);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Test: FLOAT[4] * FLOAT[4] -> FLOAT[4] (opFile name bug)\n");
    printf("Calling aclnnMulGetWorkspaceSize...\n");
    ret = (aclError)aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    printf("GetWorkspaceSize return code: %d\n", (int)ret);

    if (ret == ACL_SUCCESS) {
        void *workspace = nullptr;
        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        }

        printf("Calling aclnnMul (execution stage - kernel loading)...\n");
        ret = (aclError)aclnnMul(workspace, workspaceSize, executor, stream);
        printf("aclnnMul return code: %d\n", (int)ret);

        aclError syncRet = aclrtSynchronizeStream(stream);
        printf("Stream sync return code: %d\n", (int)syncRet);

        if (ret != ACL_SUCCESS || syncRet != ACL_SUCCESS) {
            printf("FAIL at execution: Bug confirmed - kernel file 'mul_opt' not found\n");
        } else {
            float hostOut[4] = {0};
            aclrtMemcpy(hostOut, tensorSize, devOut, tensorSize, ACL_MEMCPY_DEVICE_TO_HOST);
            printf("Results: [%.1f, %.1f, %.1f, %.1f]\n",
                   hostOut[0], hostOut[1], hostOut[2], hostOut[3]);
            printf("Expected: [5.0, 12.0, 21.0, 32.0]\n");
            printf("PASS: Mul executed successfully\n");
        }

        if (workspace) aclrtFree(workspace);
    } else {
        printf("FAIL: GetWorkspaceSize returned error %d\n", (int)ret);
        printf("Bug may manifest at workspace/tiling stage with invalid opFile\n");
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
