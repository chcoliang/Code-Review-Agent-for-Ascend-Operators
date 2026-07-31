# aclnn_mul.cpp Code Review Report

**Target Platform**: Ascend 910B, CANN 8.5.0

---

## Bug 1: `aclnnMulGetWorkspaceSize` Non-Contiguous Path Missing Platform Support Check

**位置**: 第 503-504 行

**类型**: 逻辑缺陷 / 平台兼容性

**严重程度**: 高

**描述**: 在 `aclnnMulGetWorkspaceSize` 函数中，变量 `isSupportNonContiguous`（第 476 行）仅在混合数据类型 (`isMixDataType`) 分支中使用（第 487 行），但在非混合类型分支中（第 503-504 行），当 `self` 和 `other` 的 dtype 都已等于 `promoteType` 时，直接将非连续 tensor（`selfWithStride`/`otherWithStride`）传入 `l0op::Mul`，没有检查当前平台是否支持非连续输入。在 `!IsRegBase()` 的旧平台上，Mul kernel 可能不支持非连续 tensor，导致计算错误或崩溃。

**触发输入**:
```cpp
// 在非RegBase平台（如旧固件版本）上
// self: FP32, shape=[4,4], 非连续 (stride=[8,1], storage有padding)
// other: FP32, shape=[4,4], 非连续
// out: FP32, shape=[4,4]
// 此时 promoteType = FP32 == self.dtype == other.dtype
// isMixDataType = false
// 进入第503行分支，直接使用非连续tensor调用Mul
```

**预期异常**: 在不支持非连续输入的平台上，应先调用 `l0op::Contiguous` 转为连续 tensor 再计算，或产生 kernel 执行失败错误。

**验证代码**:
```cpp
// 构造非连续FP32 tensor
int64_t shape[] = {4, 4};
int64_t strides[] = {8, 1};  // 非连续stride
auto self = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);
auto other = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 2, nullptr);
auto out = aclCreateTensor(shape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 2, nullptr);

uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
// 在非RegBase平台执行，期望正确处理非连续输入
aclnnStatus ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
// 若kernel不支持non-contiguous，执行阶段会产生错误结果
```

---

## Bug 2: `CheckInplaceMulShape` 缺少维度上限校验 (OP_CHECK_MAX_DIM)

**位置**: 第 301-306 行

**类型**: 参数校验缺失

**严重程度**: 中

**描述**: `CheckMulShape`（第 292-299 行）对 `self` 和 `other` 都调用了 `OP_CHECK_MAX_DIM(tensor, MAX_SUPPORT_DIMS_NUMS, return false)` 以防止超过最大支持维度数的 tensor 进入计算。但对称函数 `CheckInplaceMulShape` 没有进行此项校验。当 `selfRef` 或 `other` 的维度数超过 `MAX_SUPPORT_DIMS_NUMS` 时，可能导致后续 kernel 执行越界或失败。

**触发输入**:
```cpp
// 假设 MAX_SUPPORT_DIMS_NUMS = 8
// selfRef: shape=[2,2,2,2,2,2,2,2,2] (9维), dtype=FP32
// other: shape=[2] (1维, 可广播), dtype=FP32
```

**预期异常**: 应返回 `ACLNN_ERR_PARAM_INVALID`，报告维度超限。

**验证代码**:
```cpp
// 构造9维tensor
int64_t selfShape[] = {2, 2, 2, 2, 2, 2, 2, 2, 2};
int64_t otherShape[] = {2};
auto selfRef = aclCreateTensor(selfShape, 9, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, selfShape, 9, nullptr);
auto other = aclCreateTensor(otherShape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, otherShape, 1, nullptr);

uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus ret = aclnnInplaceMulGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
// 期望: ret == ACLNN_ERR_PARAM_INVALID (维度超限)
// 实际: 校验通过，后续kernel可能崩溃
```

---

## Bug 3: `aclnnInplaceMulsGetWorkspaceSize` 缺少格式校验 (MulsCheckFormat)

**位置**: 第 541-602 行（整个函数）

**类型**: 同族对称性缺失

**严重程度**: 低

**描述**: `aclnnMulsGetWorkspaceSize` 在参数检查后调用 `MulsCheckFormat(self)`（第 381 行）以检查 tensor 格式是否为 ND 并输出警告。但其 inplace 对称函数 `aclnnInplaceMulsGetWorkspaceSize` 缺少对应的格式检查调用。当输入 `selfRef` 使用非 ND 格式（如 FRACTAL_NZ）时，不会产生格式不支持的警告日志，给问题排查带来困难。

**触发输入**:
```cpp
// selfRef: dtype=FP16, format=FORMAT_FRACTAL_NZ
// other: scalar, dtype=FLOAT, value=2.0
```

**预期异常**: 应输出警告日志 "aclnnMuls only support format ND."

**验证代码**:
```cpp
int64_t shape[] = {16, 16};
auto selfRef = aclCreateTensor(shape, 2, ACL_FLOAT16, nullptr, 0, ACL_FRACTAL_NZ, shape, 2, nullptr);
auto other = aclCreateScalar(2.0f, ACL_FLOAT);

uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus ret = aclnnInplaceMulsGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
// 期望: 输出格式不支持的warning日志
// 实际: 无任何格式相关日志输出
```

---

## Bug 4: `CheckMulsParams` 缺少维度上限校验 (OP_CHECK_MAX_DIM)

**位置**: 第 308-318 行

**类型**: 参数校验缺失

**严重程度**: 中

**描述**: `CheckMulParams` 通过调用 `CheckMulShape` 间接执行了 `OP_CHECK_MAX_DIM` 检查。但 `CheckMulsParams` 仅调用 `OP_CHECK_SHAPE_NOT_EQUAL(self, out, ...)` 检查形状相等性，没有对 `self` 的维度数进行上限校验。超高维 tensor 在后续 Contiguous/Cast/Mul 操作中可能导致 kernel 越界。

**触发输入**:
```cpp
// 假设 MAX_SUPPORT_DIMS_NUMS = 8
// self: shape=[2,2,2,2,2,2,2,2,2] (9维), dtype=FP32
// other: scalar, value=3.0
// out: shape=[2,2,2,2,2,2,2,2,2] (9维), dtype=FP32
```

**预期异常**: 应返回 `ACLNN_ERR_PARAM_INVALID`。

**验证代码**:
```cpp
int64_t shape[] = {2, 2, 2, 2, 2, 2, 2, 2, 2};
auto self = aclCreateTensor(shape, 9, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 9, nullptr);
auto out = aclCreateTensor(shape, 9, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 9, nullptr);
auto other = aclCreateScalar(3.0f, ACL_FLOAT);

uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus ret = aclnnMulsGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
// 期望: ret == ACLNN_ERR_PARAM_INVALID
// 实际: 校验通过，可能kernel执行异常
```

---

## Bug 5: `aclnnInplaceMulGetWorkspaceSize` 非 RegBase 平台混合类型路径冗余 Cast

**位置**: 第 638-654 行

**类型**: 逻辑缺陷 / 同族对称性不一致

**严重程度**: 低

**描述**: 在 `aclnnInplaceMulGetWorkspaceSize` 中，`isMixDataType` 的判断结果仅在 `IsRegBase()` 为 true 时才使用（第 638 行 `if (IsRegBase() && isMixDataType)`）。而在 `aclnnMulGetWorkspaceSize` 中，混合类型分支无论 RegBase 与否都优先走混合类型路径（第 485 行 `if (isMixDataType)`）。这导致在非 RegBase 平台上，`aclnnInplaceMul` 对 (FP16, FP32) 混合类型输入会执行不必要的 Cast 到 FP32 再 Cast 回 FP16，而 `aclnnMul` 则直接利用混合类型支持。两者行为不对称。

**触发输入**:
```cpp
// 非RegBase平台
// selfRef: dtype=FP16, shape=[4,4]
// other: dtype=FP32, shape=[4,4]
```

**预期异常**: 行为应与 `aclnnMulGetWorkspaceSize` 对称，直接走混合类型路径避免冗余 Cast。

**验证代码**:
```cpp
int64_t shape[] = {4, 4};
auto selfRef = aclCreateTensor(shape, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, shape, 2, nullptr);
auto other = aclCreateTensor(shape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape, 2, nullptr);

uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus ret = aclnnInplaceMulGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
// 在非RegBase平台上，会执行额外的Cast操作
// workspaceSize比预期大（包含冗余Cast的workspace）
```

---

## 总结

| # | Bug | 位置 | 严重程度 |
|---|-----|------|----------|
| 1 | 非混合类型路径未检查平台是否支持非连续tensor | L503-504 | 高 |
| 2 | CheckInplaceMulShape 缺少 OP_CHECK_MAX_DIM | L301-306 | 中 |
| 3 | aclnnInplaceMulsGetWorkspaceSize 缺少格式检查 | L541-602 | 低 |
| 4 | CheckMulsParams 缺少 OP_CHECK_MAX_DIM | L308-318 | 中 |
| 5 | InplaceMul 非RegBase混合类型路径与Mul不对称 | L638-654 | 低 |
