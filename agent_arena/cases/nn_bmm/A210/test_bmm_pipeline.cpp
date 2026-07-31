#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    void *a, *b, *c;
    aclrtMalloc(&a, 1024*1024, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&b, 1024*1024, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c, 1024*1024, ACL_MEM_MALLOC_HUGE_FIRST);
    // launch BMM on stream...

    float h[256*256];
    aclrtMemcpy(h, sizeof(h), c, sizeof(h), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(a); aclrtFree(b); aclrtFree(c);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
