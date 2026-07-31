/**
 * Mul operator test with tensor creation loop.
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#define CHECK_RET(cond, return_expr) \
  do { if (!(cond)) { return_expr; } } while (0)

#define LOG_PRINT(message, ...) do { printf(message, ##__VA_ARGS__); } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t s = 1;
  for (auto i : shape) s *= i;
  return s;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) strides[i] = shape[i + 1] * strides[i + 1];
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  aclInit(nullptr);
  aclrtSetDevice(deviceId);
  aclrtCreateStream(&stream);

  std::vector<int64_t> shape = {4, 2};
  std::vector<float> hostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> outHostData(8, 0);

  for (int iter = 0; iter < 10000; iter++) {
    void *selfAddr = nullptr, *otherAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    CreateAclTensor(hostData, shape, &selfAddr, aclDataType::ACL_FLOAT, &self);
    CreateAclTensor(hostData, shape, &otherAddr, aclDataType::ACL_FLOAT, &other);
    CreateAclTensor(outHostData, shape, &outAddr, aclDataType::ACL_FLOAT, &out);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);


    // aclDestroyTensor(self);
    // aclDestroyTensor(other);
    // aclDestroyTensor(out);

    aclrtFree(selfAddr);
    aclrtFree(otherAddr);
    aclrtFree(outAddr);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
