#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    void *var, *m, *v, *grad, *out;
    aclrtMalloc(&var, 1024*1024*4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&m, 1024*1024*4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&v, 1024*1024*4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&grad, 1024*1024*4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&out, 1024*1024*4, ACL_MEM_MALLOC_HUGE_FIRST);

    float h[1024];
    aclrtMemcpy(h, sizeof(h), out, sizeof(h), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(var); aclrtFree(m); aclrtFree(v); aclrtFree(grad); aclrtFree(out);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
