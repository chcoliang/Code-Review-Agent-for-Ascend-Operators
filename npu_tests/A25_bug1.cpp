#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_softmax.h"
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <vector>

// Helper function to create an aclTensor
static aclTensor* create_aclTensor(std::vector<int64_t> shape, aclDataType dtype) {
    int64_t ndim = static_cast<int64_t>(shape.size());
    std::vector<int64_t> strides(ndim);
    int64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    aclTensor* tensor = aclCreateTensor(shape.data(), ndim, dtype, strides.data(), 0,
                                         ACL_FORMAT_ND, shape.data(), ndim, nullptr);
    return tensor;
}

// Bug: ASCEND910B DTYPE_SUPPORT_LIST 缺少 BF16
// 在910B平台上，BF16类型tensor会被错误拒绝
void test_bf16_rejected_on_910b() {
    aclTensor* self = create_aclTensor({2, 3}, ACL_BF16);
    aclTensor* out = create_aclTensor({2, 3}, ACL_BF16);
    int64_t dim = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor));

    // Bug验证: 910B硬件支持BF16，但代码会拒绝
    // 预期应返回成功，实际返回参数无效错误
    printf("A25 Bug1 test_bf16_rejected_on_910b: ret = %d\n", (int)ret);

    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
}

int main() {
    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", (int)initRet);
        return -1;
    }

    test_bf16_rejected_on_910b();

    aclFinalize();
    return 0;
}
