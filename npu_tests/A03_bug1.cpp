#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnn/opdev/op_errno.h"

#include <cstdio>
#include <cstdint>

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
