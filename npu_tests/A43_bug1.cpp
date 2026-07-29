#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_gelu.h"
#include <cstdio>
#include <cstdlib>

int main() {
    aclError ret = aclInit(nullptr);
    printf("aclInit: %d\n", ret);
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice: %d\n", ret);

    // GeLU: GetWorkspaceSize + Execute to verify kernel loading
    int64_t shape[] = {2, 4};
    int64_t strides[] = {4, 1};
    void *devSelf = nullptr, *devOut = nullptr;
    size_t size = 2 * 4 * 4; // FP32
    ret = aclrtMalloc(&devSelf, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc self: %d\n", ret);
    ret = aclrtMalloc(&devOut, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    printf("aclrtMalloc out: %d\n", ret);

    aclTensor *selfTensor = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                            ACL_FORMAT_ND, shape, 2, devSelf);
    aclTensor *outTensor = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                           ACL_FORMAT_ND, shape, 2, devOut);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGeluGetWorkspaceSize(selfTensor, outTensor, &workspaceSize, &executor);
    printf("aclnnGeluGetWorkspaceSize: %d, workspaceSize=%lu\n", ret, workspaceSize);

    if (ret == 0 && executor != nullptr) {
        void *workspace = nullptr;
        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        }
        aclrtStream stream = nullptr;
        aclrtCreateStream(&stream);
        ret = aclnnGelu(workspace, workspaceSize, executor, stream);
        printf("aclnnGelu: %d\n", ret);
        aclrtSynchronizeStream(stream);
        aclrtDestroyStream(stream);
        if (workspace) aclrtFree(workspace);
    }

    aclDestroyTensor(selfTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(devSelf);
    aclrtFree(devOut);
    aclrtResetDevice(0);
    aclFinalize();
    printf("Done.\n");
    return 0;
}
