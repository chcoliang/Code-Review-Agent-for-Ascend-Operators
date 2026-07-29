#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_mul.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Bug: 编译配置标志反转 DynamicCompileStaticFlag=false
// 正确行为: 正常计算，走静态tiling优化
// Buggy行为: 不走静态优化，可能性能退化或某些路径报错

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

    // Create FLOAT32 tensors shape=[1024] (static shape)
    int64_t shape[] = {1024};
    int64_t strides[] = {1};
    size_t tensorSize = 1024 * sizeof(float);

    void *devSelf = nullptr, *devOther = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc self failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOther, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc other failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOut, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc out failed: %d\n", ret); return 1; }

    // Initialize tensors
    float *hostSelf = (float *)malloc(tensorSize);
    float *hostOther = (float *)malloc(tensorSize);
    for (int i = 0; i < 1024; i++) {
        hostSelf[i] = (float)(i + 1);
        hostOther[i] = 2.0f;
    }
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

    printf("Test: FLOAT[1024] * FLOAT[1024] -> FLOAT[1024] (static shape)\n");
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

        // Read back and verify a few results
        float *hostOut = (float *)malloc(tensorSize);
        aclrtMemcpy(hostOut, tensorSize, devOut, tensorSize, ACL_MEMCPY_DEVICE_TO_HOST);
        printf("Results[0..3]: [%.1f, %.1f, %.1f, %.1f]\n",
               hostOut[0], hostOut[1], hostOut[2], hostOut[3]);
        printf("Expected[0..3]: [2.0, 4.0, 6.0, 8.0]\n");

        int pass = 1;
        for (int i = 0; i < 4; i++) {
            float expected = (float)(i + 1) * 2.0f;
            if (hostOut[i] != expected) { pass = 0; break; }
        }
        printf("%s: Static shape mul %s\n", pass ? "PASS" : "FAIL",
               pass ? "computed correctly" : "results mismatch (DynamicCompileStaticFlag bug)");

        free(hostOut);
        if (workspace) aclrtFree(workspace);
    } else {
        printf("FAIL: GetWorkspaceSize returned error %d\n", (int)ret);
        printf("Bug confirmed: DynamicCompileStaticFlag=false causes static path failure\n");
    }

    // Cleanup
    free(hostSelf);
    free(hostOther);
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
