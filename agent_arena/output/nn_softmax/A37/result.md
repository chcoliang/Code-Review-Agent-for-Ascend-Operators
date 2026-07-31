# A37 代码审查报告

## Bug: Input "x" 的第2个dtype配置为 DT_INT32，与Output "y" 的 DT_FLOAT16 不对齐，且INT32不是softmax合法输入类型

**位置**: 第23行，Input "x" 的 DataType 配置

```cpp
.DataType({ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT16})
```

**描述**:
算子注册def文件中，Input "x" 和 Output "y" 的DataType列表按位置一一对应，表示支持的输入输出dtype组合。当前配置：

| 位置 | Input x | Output y | 说明 |
|------|---------|----------|------|
| 0 | DT_FLOAT | DT_FLOAT | ✓ 正确 |
| 1 | **DT_INT32** | DT_FLOAT16 | ✗ 错误 |
| 2 | DT_BF16 | DT_BF16 | ✓ 正确 |
| 3 | DT_FLOAT16 | DT_FLOAT | ✓ half_to_float场景 |

问题：
1. **DT_INT32 不是softmax的合法输入类型**：Softmax要求浮点输入（float/float16/bf16），整数类型无法计算指数和归一化。
2. **Input-Output dtype不对齐**：第2组中 INT32→FLOAT16 无数学意义，正确应为 DT_FLOAT16→DT_FLOAT16。

正确的Input dtype列表应为：`{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16}`

**触发输入**:
- 传入 dtype=int32, shape=[2,3] 的tensor作为softmax输入

**预期异常**:
算子图编译时，GE框架会将int32输入匹配到第2组dtype配置（DT_INT32→DT_FLOAT16），算子下发到AICore执行时，kernel无法对整数数据正确执行exp/sum/div操作，导致：
- 计算结果完全错误（静默数据错误）
- 或AICore kernel执行异常（取决于kernel实现对输入dtype的处理）

**验证代码**:
```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_softmax.h"

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 创建 int32 输入 (不合法的softmax输入)
    int64_t shape[] = {2, 3};
    size_t selfSize = 2 * 3 * sizeof(int32_t);
    void *selfDevPtr;
    aclrtMalloc(&selfDevPtr, selfSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    int32_t hostData[] = {1, 2, 3, 4, 5, 6};
    aclrtMemcpy(selfDevPtr, selfSize, hostData, selfSize, ACL_MEMCPY_HOST_TO_DEVICE);
    auto selfTensor = aclCreateTensor(shape, 2, ACL_INT32, nullptr, 0,
                                       ACL_FORMAT_ND, shape, 2, selfDevPtr);

    // 创建 float16 输出
    size_t outSize = 2 * 3 * sizeof(uint16_t);
    void *outDevPtr;
    aclrtMalloc(&outDevPtr, outSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    auto outTensor = aclCreateTensor(shape, 2, ACL_FLOAT16, nullptr, 0,
                                      ACL_FORMAT_ND, shape, 2, outDevPtr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int64_t dim = 1;

    // 因def注册了INT32，框架不会拦截，算子执行产生错误结果
    auto ret = aclnnSoftmaxGetWorkspaceSize(selfTensor, dim, outTensor, &workspaceSize, &executor);
    // int32输入到达kernel层会产生计算错误

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
