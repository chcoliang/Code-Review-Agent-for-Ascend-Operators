#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    for (int i = 0; i < 100; i++) {
        int64_t s[] = {4, 8, 512, 512};
        aclTensorDesc* d = aclCreateTensorDesc(ACL_FLOAT16, 4, s, ACL_FORMAT_ND);
        aclDataBuffer* b = aclCreateDataBuffer(nullptr, 4*8*512*512*2);

    }
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
