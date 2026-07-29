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

// Bug: ASCEND910B DTYPE_SUPPORT_LIST 错误包含 DT_INT32
// Softmax是浮点运算，INT32不应被支持
void test_int32_should_be_rejected_on_910b() {
    aclTensor* self = create_aclTensor({2, 3}, ACL_INT32);
    aclTensor* out = create_aclTensor({2, 3}, ACL_INT32);
    int64_t dim = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor));

    // Bug验证: INT32不应通过softmax参数校验，但代码允许通过
    printf("A26 Bug1 test_int32_should_be_rejected: ret = %d\n", (int)ret);

    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
}

int main() {
    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", (int)initRet);
        return -1;
    }

    test_int32_should_be_rejected_on_910b();

    aclFinalize();
    return 0;
}
