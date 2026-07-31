#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    void *x, *mask, *y;
    aclrtMalloc(&x, 4*8*512*512*2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&mask, 4*8*512*512, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&y, 4*8*512*512*2, ACL_MEM_MALLOC_HUGE_FIRST);

    float h[1024];
    aclrtMemcpy(h, sizeof(h), y, sizeof(h), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(x); aclrtFree(mask); aclrtFree(y);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
