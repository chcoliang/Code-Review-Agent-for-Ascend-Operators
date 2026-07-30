#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    void *in, *out;
    aclrtMalloc(&in, 4096, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&out, 4096, ACL_MEM_MALLOC_HUGE_FIRST);
    // launch gelu on stream...
    // BUG: read without sync
    float h[1024];
    aclrtMemcpy(h, 4096, out, 4096, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(in); aclrtFree(out);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
