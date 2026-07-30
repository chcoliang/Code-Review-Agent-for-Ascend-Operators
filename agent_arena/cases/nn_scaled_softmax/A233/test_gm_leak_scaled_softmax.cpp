#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    for (int i = 0; i < 1000; i++) {
        void* ws = nullptr;
        aclrtMalloc(&ws, 16*1024*1024, ACL_MEM_MALLOC_HUGE_FIRST);
        // BUG: no free
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
