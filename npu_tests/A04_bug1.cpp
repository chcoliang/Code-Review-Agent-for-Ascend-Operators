#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnn/opdev/op_errno.h"

#include <iostream>
#include <cstdint>

// 验证 aclnnMulGetWorkspaceSize 缺少空tensor处理
// 对比 aclnnMulsGetWorkspaceSize 的行为
int main() {
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // 创建空 tensor self: shape=[0, 4], dtype=FLOAT32
    int64_t selfShape[] = {0, 4};
    int64_t selfStrides[] = {4, 1};
    aclTensor *self = aclCreateTensor(selfShape, 2, ACL_FLOAT, selfStrides, 0,
                                       ACL_FORMAT_ND, selfShape, 2, nullptr);

    // 创建 other: shape=[1, 4], dtype=FLOAT32
    int64_t otherShape[] = {1, 4};
    int64_t otherStrides[] = {4, 1};
    float otherData[] = {1.0f, 2.0f, 3.0f, 4.0f};
    void *otherDevPtr = nullptr;
    aclrtMalloc(&otherDevPtr, 4 * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(otherDevPtr, 4 * sizeof(float), otherData, 4 * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor *other = aclCreateTensor(otherShape, 2, ACL_FLOAT, otherStrides, 0,
                                        ACL_FORMAT_ND, otherShape, 2, otherDevPtr);

    // 创建空 out: shape=[0, 4], dtype=FLOAT32
    int64_t outShape[] = {0, 4};
    int64_t outStrides[] = {4, 1};
    aclTensor *out = aclCreateTensor(outShape, 2, ACL_FLOAT, outStrides, 0,
                                      ACL_FORMAT_ND, outShape, 2, nullptr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    // 调用 aclnnMulGetWorkspaceSize，对于空tensor应返回workspaceSize=0
    aclnnStatus ret = aclnnMulGetWorkspaceSize(self, other, out,
                                               &workspaceSize, &executor);

    std::cout << "=== Bug 1 验证: aclnnMulGetWorkspaceSize 缺少空tensor处理 ===" << std::endl;
    std::cout << "self shape: [0, 4] (empty tensor)" << std::endl;
    std::cout << "返回状态码: " << ret << std::endl;
    std::cout << "workspaceSize: " << workspaceSize << std::endl;
    std::cout << std::endl;
    std::cout << "正确行为: 应检测到空tensor后返回 ACLNN_SUCCESS(0) 且 workspaceSize=0，" << std::endl;
    std::cout << "         与 aclnnMulsGetWorkspaceSize/aclnnInplaceMulGetWorkspaceSize 行为一致。" << std::endl;
    std::cout << "实际行为: 未做空tensor检查，继续执行内部计算图构建，" << std::endl;
    std::cout << "         可能返回非零workspaceSize或内部错误。" << std::endl;

    // 清理
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    if (otherDevPtr) aclrtFree(otherDevPtr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
