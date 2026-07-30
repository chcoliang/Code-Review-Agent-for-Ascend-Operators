#include <acl/acl.h>
#include <iostream>

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);

    for (int i = 0; i < 100; i++) {
        int64_t shape[] = {4, 1024};
        aclTensorDesc* desc = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
        aclDataBuffer* buf = aclCreateDataBuffer(nullptr, 4 * 1024 * 2);
        // BUG: 不销毁desc和buf
        // 缺少 aclDestroyTensorDesc(desc);
        // 缺少 aclDestroyDataBuffer(buf);
    }

    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
