#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    for (int i = 0; i < 100; i++) {
        int64_t s[] = {2048, 2048};
        aclTensorDesc* d = aclCreateTensorDesc(ACL_FLOAT16, 2, s, ACL_FORMAT_ND);
        aclDataBuffer* b = aclCreateDataBuffer(nullptr, 2048*2048*2);
        // BUG: no destroy
    }
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
