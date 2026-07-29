#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_mul.h"
#include <cstdio>
#include <cstdlib>

// Bug: tiling 中 SetLocalMemorySize 用 + 代替 - (ubSize_ + DCACHE_SIZE)
// 正确行为: 正常计算
// Buggy行为: UB 超出物理容量，kernel 执行异常或崩溃

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

    // Create large FLOAT32 tensors shape=[1024,1024] to trigger tiling
    int64_t shape[] = {1024, 1024};
    int64_t strides[] = {1024, 1};
    size_t tensorSize = 1024 * 1024 * sizeof(float); // 4MB

    void *devSelf = nullptr, *devOther = nullptr, *devOut = nullptr;
    ret = aclrtMalloc(&devSelf, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc self failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOther, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc other failed: %d\n", ret); return 1; }
    ret = aclrtMalloc(&devOut, tensorSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) { printf("malloc out failed: %d\n", ret); return 1; }

    aclTensor *self = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                       ACL_FORMAT_ND, shape, 2, devSelf);
    aclTensor *other = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                        ACL_FORMAT_ND, shape, 2, devOther);
    aclTensor *out = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                      ACL_FORMAT_ND, shape, 2, devOut);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    printf("Test: FLOAT[1024,1024] * FLOAT[1024,1024] (large tensor, triggers tiling)\n");
    printf("Calling aclnnMulGetWorkspaceSize...\n");
    ret = (aclError)aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    printf("GetWorkspaceSize return code: %d\n", (int)ret);
    printf("Workspace size: %lu\n", workspaceSize);

    if (ret == ACL_SUCCESS) {
        void *workspace = nullptr;
        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        }

        printf("Calling aclnnMul (may crash if UB size exceeds physical limit)...\n");
        ret = (aclError)aclnnMul(workspace, workspaceSize, executor, stream);
        printf("aclnnMul return code: %d\n", (int)ret);

        aclError syncRet = aclrtSynchronizeStream(stream);
        printf("Stream sync return code: %d\n", (int)syncRet);

        if (ret == ACL_SUCCESS && syncRet == ACL_SUCCESS) {
            printf("PASS: Large tensor mul completed without UB overflow\n");
        } else {
            printf("FAIL: Execution error - UB size overflow bug (+ instead of -)\n");
        }

        if (workspace) aclrtFree(workspace);
    } else {
        printf("FAIL: GetWorkspaceSize returned error %d\n", (int)ret);
        printf("Bug may cause tiling to compute invalid UB size\n");
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
