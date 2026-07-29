#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnn/opdev/op_errno.h"

#include <cstdio>
#include <cstdint>

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);

    // self: shape [3,1], dtype FLOAT
    int64_t selfShape[] = {3, 1};
    int64_t selfStrides[] = {1, 1};
    float selfData[3] = {1.0f, 2.0f, 3.0f};
    aclTensor *self = aclCreateTensor(selfShape, 2, ACL_FLOAT, selfStrides, 0,
                                       ACL_FORMAT_ND, selfShape, 2, selfData);

    // other: shape [1,4], dtype FLOAT
    int64_t otherShape[] = {1, 4};
    int64_t otherStrides[] = {4, 1};
    float otherData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    aclTensor *other = aclCreateTensor(otherShape, 2, ACL_FLOAT, otherStrides, 0,
                                        ACL_FORMAT_ND, otherShape, 2, otherData);

    // out: shape [2,2] - WRONG! Should be [3,4] after broadcast
    int64_t outShape[] = {2, 2};
    int64_t outStrides[] = {2, 1};
    float outData[4] = {0};
    aclTensor *out = aclCreateTensor(outShape, 2, ACL_FLOAT, outStrides, 0,
                                      ACL_FORMAT_ND, outShape, 2, outData);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

    printf("aclnnMulGetWorkspaceSize returned: %d\n", (int)ret);
    printf("BUG: self[3,1] * other[1,4] -> broadcast shape [3,4], but out is [2,2]\n");
    if (ret == 0) {
        printf("CONFIRMED BUG: No error returned for mismatched output shape!\n");
        printf("Expected: ACLNN_ERR_PARAM_INVALID (non-zero error code)\n");
        printf("Actual: ACLNN_SUCCESS (0) - output shape validation missing\n");
    } else {
        printf("No bug: error correctly returned.\n");
    }

    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclFinalize();
    return 0;
}
