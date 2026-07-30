#include <acl/acl.h>
#include <iostream>

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // Bug: 循环分配workspace但不释放
    for (int i = 0; i < 1000; i++) {
        void* workspace = nullptr;
        aclrtMalloc(&workspace, 16 * 1024 * 1024, ACL_MEM_MALLOC_HUGE_FIRST);
        // BUG: 缺少 aclrtFree(workspace)
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
