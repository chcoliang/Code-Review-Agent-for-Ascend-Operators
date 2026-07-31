# A35 代码审查报告

## Bug: 输入tensor未做Contiguous处理直接传入SoftmaxV2算子

**位置**: 第129行，`aclnnSoftmaxGetWorkspaceSize` 函数

```cpp
auto op_out = l0op::SoftmaxV2(self, dim, uniqueExecutor.get());
```

**描述**:
代码将原始输入 `self` 直接传给 `l0op::SoftmaxV2`，而没有先调用 `l0op::Contiguous(self, ...)` 将其转为连续tensor。底层SoftmaxV2 kernel要求输入数据在内存中是连续存储的。当输入tensor是非连续的（如通过 transpose、slice、expand 等操作产生的 view tensor），直接传入会导致计算结果错误，因为kernel按连续内存布局读取数据。

**触发输入**:
- `self`: 由 transpose 产生的非连续tensor，如原始shape=[3,2]做transpose后shape=[2,3]，stride=[1,3]（非连续）
- `out`: shape=[2,3], dtype=float16（连续）
- `dim`: 1

**预期异常**:
Softmax计算结果数值错误（静默错误），因为kernel按连续布局读取了错位的数据。某些极端情况下可能产生NaN或Inf。

**验证代码**:
```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_softmax.h"

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 构造非连续tensor: 先创建[3,2]再transpose得到[2,3]的非连续view
    int64_t origShape[] = {3, 2};
    int64_t viewShape[] = {2, 3};
    int64_t strides[] = {1, 2};  // 非连续strides

    size_t dataSize = 3 * 2 * sizeof(uint16_t);  // float16
    void *devPtr;
    aclrtMalloc(&devPtr, dataSize, ACL_MEM_MALLOC_NORMAL_ONLY);

    // 创建非连续的self tensor (transpose后的view)
    auto selfTensor = aclCreateTensor(viewShape, 2, ACL_FLOAT16, strides, 0,
                                       ACL_FORMAT_ND, origShape, 2, devPtr);

    // 创建连续的out tensor
    void *outDevPtr;
    aclrtMalloc(&outDevPtr, 2 * 3 * sizeof(uint16_t), ACL_MEM_MALLOC_NORMAL_ONLY);
    auto outTensor = aclCreateTensor(viewShape, 2, ACL_FLOAT16, nullptr, 0,
                                      ACL_FORMAT_ND, viewShape, 2, outDevPtr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int64_t dim = 1;

    // 非连续输入直接传入SoftmaxV2，结果错误
    auto ret = aclnnSoftmaxGetWorkspaceSize(selfTensor, dim, outTensor, &workspaceSize, &executor);
    if (ret == ACLNN_SUCCESS) {
        void *workspace = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclnnSoftmax(workspace, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        // 此时out中数据为错误结果
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
