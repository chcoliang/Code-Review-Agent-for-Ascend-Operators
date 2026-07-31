# A38 代码审查报告

## Bug: DynamicCompileStaticFlag 配置为 false，导致动态shape场景性能劣化

**位置**: 第38行，AICore 配置

```cpp
aicoreConfig.DynamicCompileStaticFlag(false)
```

**描述**:
`DynamicCompileStaticFlag` 控制是否启用"动态编译静态化"优化。在 Ascend 910B（ascend910_95）平台上，该标志应设置为 `true`，使得动态shape算子在实际执行时可以按静态shape编译kernel，从而获得更优的tiling策略和性能。

将该标志设为 `false` 会导致：
1. 动态shape场景下，kernel始终以动态方式编译执行，无法利用具体shape信息进行编译优化
2. 对于重复出现的固定shape输入，每次都走动态路径，无法命中静态编译缓存，性能显著下降
3. 在 Ascend 910B CANN 8.5.0 环境下，SoftmaxV2 的标准配置应为 `DynamicCompileStaticFlag(true)`

**触发输入**:
- 任意合法softmax输入，如 self: shape=[batch, seq_len, hidden_dim] 动态batch场景
- 当 batch 固定为某个值反复执行时，本应触发静态编译优化但未生效

**预期异常**:
不会产生功能错误，但会导致运行时性能劣化。动态shape执行路径相比静态编译可慢数倍（具体取决于shape规模）。可通过 profiling 对比 `DynamicCompileStaticFlag(true)` 和 `false` 的kernel执行时间验证。

**验证代码**:
```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_softmax.h"
#include <chrono>
#include <cstdio>

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int64_t shape[] = {32, 128, 512};
    size_t dataSize = 32 * 128 * 512 * sizeof(uint16_t);
    void *selfDevPtr, *outDevPtr;
    aclrtMalloc(&selfDevPtr, dataSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&outDevPtr, dataSize, ACL_MEM_MALLOC_NORMAL_ONLY);

    auto selfTensor = aclCreateTensor(shape, 3, ACL_FLOAT16, nullptr, 0,
                                       ACL_FORMAT_ND, shape, 3, selfDevPtr);
    auto outTensor = aclCreateTensor(shape, 3, ACL_FLOAT16, nullptr, 0,
                                      ACL_FORMAT_ND, shape, 3, outDevPtr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int64_t dim = 2;

    // 预热
    aclnnSoftmaxGetWorkspaceSize(selfTensor, dim, outTensor, &workspaceSize, &executor);
    void *workspace = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclnnSoftmax(workspace, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    // 计时：相同shape重复执行，DynamicCompileStaticFlag=false时无法利用静态缓存
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        aclnnSoftmaxGetWorkspaceSize(selfTensor, dim, outTensor, &workspaceSize, &executor);
        aclnnSoftmax(workspace, workspaceSize, executor, stream);
    }
    aclrtSynchronizeStream(stream);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("100 iterations: %ld us (with DynamicCompileStaticFlag=false, expect slower)\n", duration);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
