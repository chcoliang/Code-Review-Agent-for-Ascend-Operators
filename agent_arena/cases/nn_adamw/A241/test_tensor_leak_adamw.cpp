#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    for (int i = 0; i < 100; i++) {
        int64_t s[] = {1024, 1024};
        aclTensorDesc* d = aclCreateTensorDesc(ACL_FLOAT, 2, s, ACL_FORMAT_ND);
        aclDataBuffer* b = aclCreateDataBuffer(nullptr, 1024*1024*4);
        // BUG: no destroy
    }
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
