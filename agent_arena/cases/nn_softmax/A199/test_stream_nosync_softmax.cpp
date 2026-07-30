#include <acl/acl.h>
#include <iostream>

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    void* devInput = nullptr;
    void* devOutput = nullptr;
    aclrtMalloc(&devInput, 1024 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devOutput, 1024 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    // Launch softmax kernel on stream...
    // BUG: 不同步就直接读取输出
    float hostOutput[1024];
    aclrtMemcpy(hostOutput, sizeof(hostOutput), devOutput, sizeof(hostOutput), ACL_MEMCPY_DEVICE_TO_HOST);
    // 缺少 aclrtSynchronizeStream(stream);

    aclrtFree(devInput);
    aclrtFree(devOutput);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
