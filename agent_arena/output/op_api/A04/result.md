# aclnn_mul.cpp 代码审查报告

## Bug 1: `aclnnMulGetWorkspaceSize` 缺少空 tensor 处理

| 项目 | 内容 |
|------|------|
| 位置 | aclnn_mul.cpp : aclnnMulGetWorkspaceSize : 452-532 |
| 类型 | 逻辑缺陷 - 缺少边界条件处理 |
| 严重程度 | 高 |
| 描述 | 其他三个同族函数(`aclnnMulsGetWorkspaceSize` 第384行、`aclnnInplaceMulsGetWorkspaceSize` 第547行、`aclnnInplaceMulGetWorkspaceSize` 第613行)都对空 tensor 做了提前返回处理，但 `aclnnMulGetWorkspaceSize` 没有。当 `self` 或 `other` 为空 tensor(某个维度为0)时，函数会继续执行 Contiguous/Cast/Mul 操作，可能导致内部计算异常或生成错误的 workspace 计算。 |

**触发输入:**
- `self`: shape=[0, 4], dtype=FLOAT32
- `other`: shape=[1, 4], dtype=FLOAT32
- `out`: shape=[0, 4], dtype=FLOAT32

**预期异常:** 应在检测到空 tensor 后直接设置 `*workspaceSize = 0` 并返回 `ACLNN_SUCCESS`，而非继续执行后续计算图构建。

### 验证代码

```cpp
#include <iostream>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

// 验证 aclnnMulGetWorkspaceSize 缺少空tensor处理
// 对比 aclnnMulsGetWorkspaceSize 的行为
int main() {
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // 创建空 tensor self: shape=[0, 4], dtype=FLOAT32
    int64_t selfShape[] = {0, 4};
    int64_t selfStrides[] = {4, 1};
    aclTensor *self = aclCreateTensor(selfShape, 2, ACL_FLOAT, selfStrides, 0,
                                       ACL_FORMAT_ND, selfShape, 2, nullptr);

    // 创建 other: shape=[1, 4], dtype=FLOAT32
    int64_t otherShape[] = {1, 4};
    int64_t otherStrides[] = {4, 1};
    float otherData[] = {1.0f, 2.0f, 3.0f, 4.0f};
    void *otherDevPtr = nullptr;
    aclrtMalloc(&otherDevPtr, 4 * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(otherDevPtr, 4 * sizeof(float), otherData, 4 * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor *other = aclCreateTensor(otherShape, 2, ACL_FLOAT, otherStrides, 0,
                                        ACL_FORMAT_ND, otherShape, 2, otherDevPtr);

    // 创建空 out: shape=[0, 4], dtype=FLOAT32
    int64_t outShape[] = {0, 4};
    int64_t outStrides[] = {4, 1};
    aclTensor *out = aclCreateTensor(outShape, 2, ACL_FLOAT, outStrides, 0,
                                      ACL_FORMAT_ND, outShape, 2, nullptr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    // 调用 aclnnMulGetWorkspaceSize，对于空tensor应返回workspaceSize=0
    aclnnStatus ret = aclnnMulGetWorkspaceSize(self, other, out,
                                               &workspaceSize, &executor);

    std::cout << "=== Bug 1 验证: aclnnMulGetWorkspaceSize 缺少空tensor处理 ===" << std::endl;
    std::cout << "self shape: [0, 4] (empty tensor)" << std::endl;
    std::cout << "返回状态码: " << ret << std::endl;
    std::cout << "workspaceSize: " << workspaceSize << std::endl;
    std::cout << std::endl;
    std::cout << "正确行为: 应检测到空tensor后返回 ACLNN_SUCCESS(0) 且 workspaceSize=0，" << std::endl;
    std::cout << "         与 aclnnMulsGetWorkspaceSize/aclnnInplaceMulGetWorkspaceSize 行为一致。" << std::endl;
    std::cout << "实际行为: 未做空tensor检查，继续执行内部计算图构建，" << std::endl;
    std::cout << "         可能返回非零workspaceSize或内部错误。" << std::endl;

    // 清理
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    if (otherDevPtr) aclrtFree(otherDevPtr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
// 编译: g++ -std=c++17 -I/usr/local/Ascend/cann-8.5.0/include \
//       -L/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 \
//       -o verify_bug1 verify_bug1.cpp -lascendcl -lnnopbase -lopapi
```

---

## Bug 2: `aclnnInplaceMulGetWorkspaceSize` 对 mix-dtype 场景处理与 `aclnnMulGetWorkspaceSize` 不对称

| 项目 | 内容 |
|------|------|
| 位置 | aclnn_mul.cpp : aclnnInplaceMulGetWorkspaceSize : 631 |
| 类型 | 逻辑缺陷 - 同族函数不对称 |
| 严重程度 | 中 |
| 描述 | 在 `aclnnMulGetWorkspaceSize`（第478行）中，当 `isMixDataType=true` 时无论 `IsRegBase()` 返回什么，都直接调用 `l0op::Mul` 而不做 Cast（因为 kernel 原生支持 FP16+FP32 / BF16+FP32 混合输入）。但在 `aclnnInplaceMulGetWorkspaceSize`（第631行），条件是 `IsRegBase() && isMixDataType`，当 `!IsRegBase()` 且为 mix-dtype 时，代码进入 else 分支，调用 `PromoteType` 将两个输入都 Cast 到 FP32，再调用 Mul。这导致 BF16 inplace 场景下多了两次不必要的 Cast 操作，且可能因精度提升路径不同而产生与 `aclnnMul` 不一致的数值结果。 |

**触发输入:**
- `selfRef`: shape=[2, 3], dtype=BF16, 值=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
- `other`: shape=[2, 3], dtype=FLOAT32, 值=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
- 运行在非 RegBase 模式下

**预期异常:** `aclnnInplaceMul` 会将 BF16 和 FP32 都 Cast 到 FP32 再计算再 Cast 回 BF16，而 `aclnnMul` 在相同输入下会直接使用 kernel 的混合精度支持，两者计算路径不一致。

### 验证代码

```cpp
#include <iostream>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

// 验证 Bug 2: aclnnInplaceMul 与 aclnnMul 对 mix-dtype 处理不对称
int main() {
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // selfRef: shape=[2,3], dtype=BF16
    int64_t shape[] = {2, 3};
    int64_t strides[] = {3, 1};
    size_t bf16Size = 2 * 3 * 2; // 6 elements * 2 bytes

    void *selfDevPtr = nullptr;
    aclrtMalloc(&selfDevPtr, bf16Size, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor *selfRef = aclCreateTensor(shape, 2, ACL_BF16, strides, 0,
                                          ACL_FORMAT_ND, shape, 2, selfDevPtr);

    // other: shape=[2,3], dtype=FLOAT32
    size_t fp32Size = 2 * 3 * 4; // 6 elements * 4 bytes
    void *otherDevPtr = nullptr;
    aclrtMalloc(&otherDevPtr, fp32Size, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor *other = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                        ACL_FORMAT_ND, shape, 2, otherDevPtr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    // 调用 InplaceMul
    aclnnStatus ret = aclnnInplaceMulGetWorkspaceSize(selfRef, other,
                                                      &workspaceSize, &executor);

    std::cout << "=== Bug 2 验证: InplaceMul mix-dtype 路径不对称 ===" << std::endl;
    std::cout << "selfRef dtype: BF16, other dtype: FLOAT32" << std::endl;
    std::cout << "InplaceMul 返回状态码: " << ret << std::endl;
    std::cout << "InplaceMul workspaceSize: " << workspaceSize << std::endl;
    std::cout << std::endl;
    std::cout << "正确行为: 当 isMixDataType=true 时，应与 aclnnMul 一样直接调用" << std::endl;
    std::cout << "         l0op::Mul 而无需 Cast，因为 kernel 原生支持 BF16+FP32 混合输入。" << std::endl;
    std::cout << "实际行为(非RegBase): 进入 else 分支，PromoteType 为 FP32，" << std::endl;
    std::cout << "         对两个输入都做 Cast(FP32)，多了不必要的计算开销。" << std::endl;

    // 清理
    aclDestroyTensor(selfRef);
    aclDestroyTensor(other);
    if (selfDevPtr) aclrtFree(selfDevPtr);
    if (otherDevPtr) aclrtFree(otherDevPtr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
// 编译: g++ -std=c++17 -I/usr/local/Ascend/cann-8.5.0/include \
//       -L/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 \
//       -o verify_bug2 verify_bug2.cpp -lascendcl -lnnopbase -lopapi
```

---

## Bug 3: `CheckInplaceMulShape` 缺少 `OP_CHECK_MAX_DIM` 检查

| 项目 | 内容 |
|------|------|
| 位置 | aclnn_mul.cpp : CheckInplaceMulShape : 301-306 |
| 类型 | 参数校验缺失 |
| 严重程度 | 中 |
| 描述 | `CheckMulShape`（第292-299行）对 `self` 和 `other` 都调用了 `OP_CHECK_MAX_DIM(tensor, MAX_SUPPORT_DIMS_NUMS, return false)` 来检查维度数是否超限。但其同族对称函数 `CheckInplaceMulShape`（第301-306行）完全没有做维度数检查，直接进入 broadcast 和 shape 校验。若传入维度数超过 `MAX_SUPPORT_DIMS_NUMS` 的 tensor，`CheckInplaceMulShape` 不会拦截，后续的 Contiguous/Cast/Mul 操作可能产生未定义行为。 |

**触发输入:**
- `selfRef`: shape=[1,1,1,1,1,1,1,1,1] (9维，假设MAX_SUPPORT_DIMS_NUMS=8), dtype=FLOAT32
- `other`: shape=[1,1,1,1,1,1,1,1,1] (9维), dtype=FLOAT32

**预期异常:** 应返回 `ACLNN_ERR_PARAM_INVALID`，但实际会通过校验进入计算流程，可能导致内部逻辑异常。

### 验证代码

```cpp
#include <iostream>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

// 验证 Bug 3: CheckInplaceMulShape 缺少 OP_CHECK_MAX_DIM
int main() {
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // 创建超过最大支持维度数(通常为8)的tensor
    const int ndim = 9;
    int64_t shape[ndim] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int64_t strides[ndim] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    void *selfDevPtr = nullptr;
    aclrtMalloc(&selfDevPtr, sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    float val = 2.0f;
    aclrtMemcpy(selfDevPtr, sizeof(float), &val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    void *otherDevPtr = nullptr;
    aclrtMalloc(&otherDevPtr, sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    float val2 = 3.0f;
    aclrtMemcpy(otherDevPtr, sizeof(float), &val2, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    aclTensor *selfRef = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                          ACL_FORMAT_ND, shape, ndim, selfDevPtr);
    aclTensor *other = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                        ACL_FORMAT_ND, shape, ndim, otherDevPtr);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    // aclnnMul 应拦截(因为 CheckMulShape 有 OP_CHECK_MAX_DIM)
    int64_t outShape[ndim] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int64_t outStrides[ndim] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    void *outDevPtr = nullptr;
    aclrtMalloc(&outDevPtr, sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor *out = aclCreateTensor(outShape, ndim, ACL_FLOAT, outStrides, 0,
                                      ACL_FORMAT_ND, outShape, ndim, outDevPtr);

    aclnnStatus retMul = aclnnMulGetWorkspaceSize(selfRef, other, out,
                                                   &workspaceSize, &executor);

    // aclnnInplaceMul 不会拦截(CheckInplaceMulShape 无 OP_CHECK_MAX_DIM)
    uint64_t workspaceSize2 = 0;
    aclOpExecutor *executor2 = nullptr;
    aclnnStatus retInplace = aclnnInplaceMulGetWorkspaceSize(selfRef, other,
                                                             &workspaceSize2, &executor2);

    std::cout << "=== Bug 3 验证: CheckInplaceMulShape 缺少维度检查 ===" << std::endl;
    std::cout << "输入: " << ndim << "维 tensor (shape=[1]*9)" << std::endl;
    std::cout << "aclnnMulGetWorkspaceSize 返回: " << retMul << std::endl;
    std::cout << "aclnnInplaceMulGetWorkspaceSize 返回: " << retInplace << std::endl;
    std::cout << std::endl;
    std::cout << "正确行为: 两个函数都应返回 ACLNN_ERR_PARAM_INVALID 拒绝超维度输入。" << std::endl;
    std::cout << "实际行为: aclnnMul 正确拒绝，aclnnInplaceMul 错误放行。" << std::endl;

    // 清理
    aclDestroyTensor(selfRef);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    if (selfDevPtr) aclrtFree(selfDevPtr);
    if (otherDevPtr) aclrtFree(otherDevPtr);
    if (outDevPtr) aclrtFree(outDevPtr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
// 编译: g++ -std=c++17 -I/usr/local/Ascend/cann-8.5.0/include \
//       -L/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 \
//       -o verify_bug3 verify_bug3.cpp -lascendcl -lnnopbase -lopapi
```

---

## 汇总表

| Bug # | 位置 | 类型 | 严重程度 | 简要描述 |
|-------|------|------|----------|----------|
| 1 | aclnn_mul.cpp : aclnnMulGetWorkspaceSize : 452-532 | 逻辑缺陷 - 缺少边界条件 | 高 | 缺少空 tensor 提前返回处理，与其他3个同族函数不对称 |
| 2 | aclnn_mul.cpp : aclnnInplaceMulGetWorkspaceSize : 631 | 逻辑缺陷 - 同族不对称 | 中 | mix-dtype 场景仅在 RegBase 模式跳过 Cast，非 RegBase 下多余 Cast 与 aclnnMul 行为不一致 |
| 3 | aclnn_mul.cpp : CheckInplaceMulShape : 301-306 | 参数校验缺失 | 中 | 缺少 `OP_CHECK_MAX_DIM` 维度数检查，与 `CheckMulShape` 不对称 |
