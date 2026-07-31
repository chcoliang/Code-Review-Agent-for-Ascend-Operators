#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    for (int i = 0; i < 100; i++) {
        int64_t s[] = {32, 1024};
        aclTensorDesc* d = aclCreateTensorDesc(ACL_FLOAT16, 2, s, ACL_FORMAT_ND);
        aclDataBuffer* b = aclCreateDataBuffer(nullptr, 32*1024*2);

    }
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
