#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_softmax.h"
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <vector>

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

// Bug: CheckShape 缺少维度上限校验（AXIS_LIMIT=8）
// 9维tensor应被拒绝但实际通过校验
void test_9d_tensor_should_be_rejected() {
    aclTensor* self = create_aclTensor({2,2,2,2,2,2,2,2,2}, ACL_FLOAT);  // 9维
    aclTensor* out = create_aclTensor({2,2,2,2,2,2,2,2,2}, ACL_FLOAT);   // 9维
    int64_t dim = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor));

    // Bug验证: 缺少维度上限检查，9维tensor不会被拒绝
    printf("A28 Bug1 test_9d_tensor: ret = %d (expected ACLNN_ERR_PARAM_INVALID)\n", (int)ret);

    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
}

void test_8d_tensor_should_pass() {
    aclTensor* self = create_aclTensor({2,2,2,2,2,2,2,2}, ACL_FLOAT);  // 8维
    aclTensor* out = create_aclTensor({2,2,2,2,2,2,2,2}, ACL_FLOAT);   // 8维
    int64_t dim = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor));

    // 8维应该通过
    printf("A28 Bug1 test_8d_tensor: ret = %d (expected ACLNN_SUCCESS)\n", (int)ret);

    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
}

int main() {
    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", (int)initRet);
        return -1;
    }

    test_9d_tensor_should_be_rejected();
    test_8d_tensor_should_pass();

    aclFinalize();
    return 0;
}
