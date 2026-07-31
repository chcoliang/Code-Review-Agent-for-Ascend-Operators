# Code Review: aclnn_mul.cpp (A03)

## 审查结果

### Bug 1: aclnnMulGetWorkspaceSize 错误拒绝 DT_DOUBLE 类型的 self 输入
- **位置**: aclnn_mul.cpp:aclnnMulGetWorkspaceSize:466
- **类型**: 逻辑错误 / 对称性破坏
- **严重程度**: 高
- **描述**: 在 `aclnnMulGetWorkspaceSize` 第466行存在一个硬编码检查 `if (self->GetDataType() == DataType::DT_DOUBLE) return ACLNN_ERR_PARAM_INVALID;`。然而 `DT_DOUBLE` 明确列在 `ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST` 和 `ASCEND910_DTYPE_DTYPE_SUPPORT_LIST` 中，且在第324行 `CheckMulDtype` 校验中已通过。该检查存在两个问题：(1) 与声明的dtype支持列表矛盾，导致API行为不一致；(2) 只检查 `self` 不检查 `other`，破坏了乘法交换律——`Mul(DOUBLE_tensor, INT32_tensor)` 失败，但 `Mul(INT32_tensor, DOUBLE_tensor)` 成功。
- **触发输入**: `self: shape=[2,3], dtype=DT_DOUBLE, values=[[1.0,2.0,3.0],[4.0,5.0,6.0]]`; `other: shape=[2,3], dtype=DT_DOUBLE, values=[[1.0,1.0,1.0],[1.0,1.0,1.0]]`; `out: shape=[2,3], dtype=DT_DOUBLE`
- **预期异常**: 函数返回 `ACLNN_ERR_PARAM_INVALID`（错误码），但按照支持列表和参数校验逻辑，应返回 `ACLNN_SUCCESS` 并正常计算

#### 验证代码
```cpp
// 编译: g++ -std=c++17 -o test test.cpp -I/usr/local/Ascend/cann-8.5.0/include -L/usr/local/Ascend/cann-8.5.0/lib64 -L/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 -lascendcl -lnnopbase -lopapi -Wl,-rpath,/usr/local/Ascend/cann-8.5.0/lib64:/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 -Wl,--allow-shlib-undefined
#include <cstdio>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

// 验证 Bug: aclnnMulGetWorkspaceSize 错误拒绝 DT_DOUBLE self 输入
// 预期正确行为: DT_DOUBLE 在支持列表中，应返回 ACLNN_SUCCESS (0)
// 实际行为: 返回 ACLNN_ERR_PARAM_INVALID

int main() {
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);

    // 创建 self tensor: shape=[2,3], dtype=DOUBLE
    int64_t selfShape[] = {2, 3};
    int64_t selfStrides[] = {3, 1};
    double selfData[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    void* selfDevData = nullptr;
    aclrtMalloc(&selfDevData, 6 * sizeof(double), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(selfDevData, 6 * sizeof(double), selfData, 6 * sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
    
    aclTensor* self = aclCreateTensor(selfShape, 2, ACL_DOUBLE, selfStrides, 0,
                                       ACL_FORMAT_ND, selfShape, 2, selfDevData);

    // 创建 other tensor: shape=[2,3], dtype=DOUBLE
    double otherData[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    void* otherDevData = nullptr;
    aclrtMalloc(&otherDevData, 6 * sizeof(double), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(otherDevData, 6 * sizeof(double), otherData, 6 * sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
    
    aclTensor* other = aclCreateTensor(selfShape, 2, ACL_DOUBLE, selfStrides, 0,
                                        ACL_FORMAT_ND, selfShape, 2, otherDevData);

    // 创建 out tensor: shape=[2,3], dtype=DOUBLE
    void* outDevData = nullptr;
    aclrtMalloc(&outDevData, 6 * sizeof(double), ACL_MEM_MALLOC_NORMAL_ONLY);
    
    aclTensor* out = aclCreateTensor(selfShape, 2, ACL_DOUBLE, selfStrides, 0,
                                      ACL_FORMAT_ND, selfShape, 2, outDevData);

    // 测试1: self=DOUBLE (应成功但会失败)
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    
    printf("=== Bug 验证: aclnnMulGetWorkspaceSize 错误拒绝 DT_DOUBLE ===\n");
    printf("输入: self=DOUBLE[2,3], other=DOUBLE[2,3], out=DOUBLE[2,3]\n");
    printf("实际返回值: %d\n", (int)ret);
    printf("正确行为: 应返回 ACLNN_SUCCESS (0), 因为 DT_DOUBLE 在支持列表中\n");
    printf("Bug原因: 第466行硬编码 if(self->GetDataType()==DT_DOUBLE) return ACLNN_ERR_PARAM_INVALID\n\n");

    // 测试2: 交换律破坏验证 - self=INT32, other=DOUBLE (应成功)
    int32_t intData[] = {1, 2, 3, 4, 5, 6};
    void* intDevData = nullptr;
    aclrtMalloc(&intDevData, 6 * sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(intDevData, 6 * sizeof(int32_t), intData, 6 * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    
    int64_t intStrides[] = {3, 1};
    aclTensor* selfInt = aclCreateTensor(selfShape, 2, ACL_INT32, intStrides, 0,
                                          ACL_FORMAT_ND, selfShape, 2, intDevData);

    aclOpExecutor* executor2 = nullptr;
    uint64_t workspaceSize2 = 0;
    // self=INT32, other=DOUBLE -> 应该能正常工作(self不是DOUBLE，不触发466行)
    aclnnStatus ret2 = aclnnMulGetWorkspaceSize(selfInt, other, out, &workspaceSize2, &executor2);
    
    printf("=== 交换律破坏验证 ===\n");
    printf("Mul(INT32, DOUBLE) 返回: %d\n", (int)ret2);
    
    // self=DOUBLE, other=INT32 -> 会被466行拒绝
    aclOpExecutor* executor3 = nullptr;
    uint64_t workspaceSize3 = 0;
    aclnnStatus ret3 = aclnnMulGetWorkspaceSize(self, selfInt, out, &workspaceSize3, &executor3);
    printf("Mul(DOUBLE, INT32) 返回: %d\n", (int)ret3);
    printf("正确行为: 两者都应返回相同结果(乘法交换律)\n");
    printf("实际行为: Mul(DOUBLE,INT32)被错误拒绝, 交换律被破坏\n");

    // 清理
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclDestroyTensor(selfInt);
    aclrtFree(selfDevData);
    aclrtFree(otherDevData);
    aclrtFree(outDevData);
    aclrtFree(intDevData);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

---

## 汇总表

| Bug编号 | 位置 | 触发条件 | 预期异常 |
|---------|------|----------|----------|
| 1 | aclnn_mul.cpp:aclnnMulGetWorkspaceSize:466 | self 的 dtype 为 DT_DOUBLE（如 shape=[2,3], dtype=DOUBLE） | 错误返回 ACLNN_ERR_PARAM_INVALID；破坏乘法交换律（self=DOUBLE 失败，other=DOUBLE 成功） |
