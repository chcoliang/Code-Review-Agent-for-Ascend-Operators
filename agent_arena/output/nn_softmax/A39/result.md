# A39 代码审查报告

## Bug: opFile.value 配置错误，"softmax_v2_opt" 应为 "softmax_v2_apt"

**位置**: 第44行，ExtendCfgInfo 配置

```cpp
.ExtendCfgInfo("opFile.value", "softmax_v2_opt");
```

**描述**:
`opFile.value` 指定了算子kernel的二进制文件名（即APT编译产物的文件名）。SoftmaxV2 算子使用 Ascend Programming Template (APT) 框架开发，其编译产物文件名应为 `"softmax_v2_apt"`。当前配置错误地写为 `"softmax_v2_opt"`（将 "apt" 误写为 "opt"）。

这会导致运行时框架根据错误的文件名查找kernel二进制文件，找不到对应的 `.o` 或 `.json` 文件，算子无法正确加载和执行。

**触发输入**:
- 任意合法的softmax调用，如 self: shape=[2,3], dtype=float16, dim=1

**预期异常**:
算子执行时框架报错，无法找到名为 `"softmax_v2_opt"` 的kernel文件。典型错误信息类似：
- `[ERROR] GE: Failed to find op file: softmax_v2_opt`
- 返回错误码，如 `ACL_ERROR_GE_INTERNAL_ERROR` 或 `ACLNN_ERR_INNER_NULLPTR`
- 算子图编译/加载阶段即失败

**验证代码**:
```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_softmax.h"
#include <cstdio>

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int64_t shape[] = {2, 3};
    size_t dataSize = 2 * 3 * sizeof(uint16_t);
    void *selfDevPtr, *outDevPtr;
    aclrtMalloc(&selfDevPtr, dataSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&outDevPtr, dataSize, ACL_MEM_MALLOC_NORMAL_ONLY);

    auto selfTensor = aclCreateTensor(shape, 2, ACL_FLOAT16, nullptr, 0,
                                       ACL_FORMAT_ND, shape, 2, selfDevPtr);
    auto outTensor = aclCreateTensor(shape, 2, ACL_FLOAT16, nullptr, 0,
                                      ACL_FORMAT_ND, shape, 2, outDevPtr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int64_t dim = 1;

    auto ret = aclnnSoftmaxGetWorkspaceSize(selfTensor, dim, outTensor, &workspaceSize, &executor);
    printf("GetWorkspaceSize ret: %d\n", ret);

    if (ret == ACLNN_SUCCESS && executor != nullptr) {
        void *workspace = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        // 此处执行时将因找不到 "softmax_v2_opt" kernel文件而失败
        auto execRet = aclnnSoftmax(workspace, workspaceSize, executor, stream);
        printf("Execute ret: %d (expect failure due to wrong opFile name)\n", execRet);
        aclrtSynchronizeStream(stream);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
