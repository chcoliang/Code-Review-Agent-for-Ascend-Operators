#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    for (int i = 0; i < 100; i++) {
        int64_t s[] = {1, 64, 112, 112};
        aclTensorDesc* d = aclCreateTensorDesc(ACL_FLOAT, 4, s, ACL_FORMAT_NCHW);
        aclDataBuffer* b = aclCreateDataBuffer(nullptr, 1*64*112*112*4);
        // BUG: no destroy
    }
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
