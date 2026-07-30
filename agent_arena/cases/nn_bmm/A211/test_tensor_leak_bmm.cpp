#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    for (int i = 0; i < 100; i++) {
        int64_t s[] = {4, 128, 128};
        aclTensorDesc* d = aclCreateTensorDesc(ACL_FLOAT16, 3, s, ACL_FORMAT_ND);
        aclDataBuffer* b = aclCreateDataBuffer(nullptr, 4*128*128*2);
        // BUG: no destroy
    }
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
