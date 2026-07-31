#include <acl/acl.h>
int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    void *in, *weight, *out;
    aclrtMalloc(&in, 1*3*224*224*4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&weight, 64*3*7*7*4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&out, 1*64*112*112*4, ACL_MEM_MALLOC_HUGE_FIRST);
    // launch conv on stream...

    float h[1024];
    aclrtMemcpy(h, sizeof(h), out, sizeof(h), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(in); aclrtFree(weight); aclrtFree(out);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
