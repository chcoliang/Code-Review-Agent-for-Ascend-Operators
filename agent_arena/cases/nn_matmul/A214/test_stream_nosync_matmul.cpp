#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    void *a, *b, *c;
    aclrtMalloc(&a, 2048*2048*2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&b, 2048*2048*2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c, 2048*2048*2, ACL_MEM_MALLOC_HUGE_FIRST);
    // launch MatMul on stream...

    float h[1024];
    aclrtMemcpy(h, sizeof(h), c, sizeof(h), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(a); aclrtFree(b); aclrtFree(c);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
