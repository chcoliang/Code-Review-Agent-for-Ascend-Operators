#include <acl/acl.h>
#include <iostream>

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    for (int i = 0; i < 1000; i++) {
        void* workspace = nullptr;
        aclrtMalloc(&workspace, 16 * 1024 * 1024, ACL_MEM_MALLOC_HUGE_FIRST);

    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
