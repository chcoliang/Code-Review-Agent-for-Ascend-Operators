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

// Bug: 空指针校验失败时返回 ACLNN_SUCCESS 而非错误码
void test_null_self_returns_success_instead_of_error() {
    const aclTensor* self = nullptr;
    aclTensor* out = create_aclTensor({2, 3}, ACL_FLOAT);
    int64_t dim = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor));

    // Bug验证: 空指针时返回ACLNN_SUCCESS而非ACLNN_ERR_PARAM_NULLPTR
    // 可能导致后续空指针解引用崩溃
    printf("A27 Bug1 test_null_self: ret = %d (expected ACLNN_ERR_PARAM_NULLPTR)\n", (int)ret);

    if (out) aclDestroyTensor(out);
}

void test_null_out_returns_success_instead_of_error() {
    aclTensor* self = create_aclTensor({2, 3}, ACL_FLOAT);
    aclTensor* out = nullptr;
    int64_t dim = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor));

    printf("A27 Bug1 test_null_out: ret = %d (expected ACLNN_ERR_PARAM_NULLPTR)\n", (int)ret);

    if (self) aclDestroyTensor(self);
}

int main() {
    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", (int)initRet);
        return -1;
    }

    test_null_self_returns_success_instead_of_error();
    test_null_out_returns_success_instead_of_error();

    aclFinalize();
    return 0;
}
