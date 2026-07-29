#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_gelu.h"
#include <cassert>
#include <cstdio>
#include <cstdint>

// Bug: CheckShape函数未校验self和out的shape一致性
// (void)out 直接忽略了out参数，缺少OP_CHECK_SHAPE_NOT_EQUAL校验
void test_shape_mismatch_not_detected() {
    int64_t self_shape[] = {2, 3};
    int64_t self_strides[] = {3, 1};
    int64_t out_shape[] = {3, 2};
    int64_t out_strides[] = {2, 1};
    aclTensor *self = aclCreateTensor(self_shape, 2, ACL_FLOAT, self_strides, 0,
                                       ACL_FORMAT_ND, self_shape, 2, nullptr);
    aclTensor *out = aclCreateTensor(out_shape, 2, ACL_FLOAT, out_strides, 0,
                                      ACL_FORMAT_ND, out_shape, 2, nullptr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor));

    // 预期: 应返回 ACLNN_ERR_PARAM_INVALID, 因为self和out的shape不一致
    // 实际: 由于CheckShape未校验out, 返回ACLNN_SUCCESS
    printf("A33 Bug1 test_shape_mismatch: ret = %d (expected ACLNN_ERR_PARAM_INVALID)\n", (int)ret);

    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
}

void test_different_ndim_not_detected() {
    int64_t self_shape[] = {2, 3};
    int64_t self_strides[] = {3, 1};
    int64_t out_shape[] = {6};
    int64_t out_strides[] = {1};
    aclTensor *self = aclCreateTensor(self_shape, 2, ACL_FLOAT, self_strides, 0,
                                       ACL_FORMAT_ND, self_shape, 2, nullptr);
    aclTensor *out = aclCreateTensor(out_shape, 1, ACL_FLOAT, out_strides, 0,
                                      ACL_FORMAT_ND, out_shape, 1, nullptr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclError ret = static_cast<aclError>(aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor));

    // 预期: 应返回 ACLNN_ERR_PARAM_INVALID
    // 实际: 校验通过, 可能导致内存错误
    printf("A33 Bug1 test_different_ndim: ret = %d (expected ACLNN_ERR_PARAM_INVALID)\n", (int)ret);

    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
}

int main() {
    aclError initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        printf("aclInit failed: %d\n", (int)initRet);
        return -1;
    }

    test_shape_mismatch_not_detected();
    test_different_ndim_not_detected();

    aclFinalize();
    return 0;
}
