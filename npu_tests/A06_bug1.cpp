#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnn/opdev/op_errno.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>

int main() {
    // 初始化ACL
    aclInit(nullptr);
    
    // 创建有效的输入tensor: shape=[2,3], dtype=FLOAT
    int64_t shape[] = {2, 3};
    int64_t strides[] = {3, 1};
    aclTensor *self = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);
    aclTensor *out = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);
    
    float scalarVal = 2.0f;
    aclScalar *other = aclCreateScalar(&scalarVal, ACL_FLOAT);
    
    aclOpExecutor *executor = nullptr;
    
    printf("Bug 1: Calling aclnnMulsGetWorkspaceSize with workspaceSize=NULL\n");
    printf("Expected: should return ACLNN_ERR_PARAM_NULLPTR (161001)\n");
    printf("Actual: will SEGFAULT due to dereferencing NULL workspaceSize pointer\n");
    fflush(stdout);
    
    // 传入 workspaceSize = NULL，应触发 SEGFAULT
    aclnnStatus ret = aclnnMulsGetWorkspaceSize(self, other, out,
                                                 nullptr,  // workspaceSize = NULL!
                                                 &executor);
    
    // 如果能到这里说明 bug 已修复
    printf("Return code: %d (should not reach here if bug exists)\n", ret);
    
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclDestroyScalar(other);
    aclFinalize();
    return 0;
}
