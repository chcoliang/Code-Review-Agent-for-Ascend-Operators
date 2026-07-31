# A36 代码审查报告

## Bug: 输入tensor未做Contiguous处理直接传入Gelu算子

**位置**: 第104行，`aclnnGeluGetWorkspaceSize` 函数

```cpp
auto geluResult = l0op::Gelu(self, uniqueExecutor.get());
```

**描述**:
代码将原始输入 `self` 直接传给 `l0op::Gelu`，没有先调用 `l0op::Contiguous(self, uniqueExecutor.get())` 将输入转为连续tensor。虽然头文件中包含了 `"aclnn_kernels/contiguous.h"`，但实际并未使用。底层Gelu kernel要求输入在内存中连续存储。当输入为非连续tensor时（如经过slice、transpose、permute等操作），kernel会按连续布局读取内存，导致计算结果错误。

**触发输入**:
- `self`: 非连续tensor，例如对shape=[4,4]的tensor做 `[:, ::2]` 切片后得到shape=[4,2], stride=[4,2]的非连续tensor
- `out`: shape=[4,2], dtype=float16（连续）

**预期异常**:
Gelu计算结果数值错误（静默错误）。Kernel按连续内存布局读取了错位的数据，输出结果与预期不符。

**验证代码**:
```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_gelu.h"

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 构造非连续tensor: 创建[4,4]后取[:, ::2]得到[4,2]非连续view
    int64_t storageShape[] = {4, 4};
    int64_t viewShape[] = {4, 2};
    int64_t strides[] = {4, 2};  // stride[1]=2 != 1, 非连续

    size_t dataSize = 4 * 4 * sizeof(uint16_t);  // float16
    void *devPtr;
    aclrtMalloc(&devPtr, dataSize, ACL_MEM_MALLOC_NORMAL_ONLY);

    // 用float16数据初始化
    uint16_t hostData[16];
    for (int i = 0; i < 16; i++) hostData[i] = 0x3C00; // 1.0 in fp16
    aclrtMemcpy(devPtr, dataSize, hostData, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 创建非连续的self tensor
    auto selfTensor = aclCreateTensor(viewShape, 2, ACL_FLOAT16, strides, 0,
                                       ACL_FORMAT_ND, storageShape, 2, devPtr);

    // 创建连续的out tensor
    void *outDevPtr;
    size_t outSize = 4 * 2 * sizeof(uint16_t);
    aclrtMalloc(&outDevPtr, outSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    auto outTensor = aclCreateTensor(viewShape, 2, ACL_FLOAT16, nullptr, 0,
                                      ACL_FORMAT_ND, viewShape, 2, outDevPtr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnGeluGetWorkspaceSize(selfTensor, outTensor, &workspaceSize, &executor);
    if (ret == ACLNN_SUCCESS) {
        void *workspace = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclnnGelu(workspace, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        // 输出结果错误：非连续数据被错误解读
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
