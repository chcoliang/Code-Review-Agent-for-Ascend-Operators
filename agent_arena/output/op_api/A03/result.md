# aclnn_mul.cpp 代码审查报告

## Bug 列表

### Bug 1: aclnnMulGetWorkspaceSize 中对 DT_DOUBLE 的错误硬编码拒绝

- **位置**: 第 466 行
- **类型**: 逻辑错误 / 不一致约束
- **严重程度**: 高
- **描述**: 在 `aclnnMulGetWorkspaceSize` 中，`if (self->GetDataType() == DataType::DT_DOUBLE) return ACLNN_ERR_PARAM_INVALID;` 硬编码拒绝了 `self` 为 DT_DOUBLE 的情况。然而 DT_DOUBLE 在 `ASCEND910_DTYPE_DTYPE_SUPPORT_LIST` 和 `ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST` 中均存在（第 54、59 行），已经通过了 `CheckMulDtype` 的验证。此外该检查仅验证 `self`，不验证 `other` 为 DT_DOUBLE 的情况，逻辑不完整。如果确实需要限制 DT_DOUBLE，应从支持列表中移除或在参数检查阶段统一处理，而非在通过校验后额外硬拦截。
- **触发条件**: 调用 `aclnnMulGetWorkspaceSize` 时，`self` tensor 的数据类型为 DT_DOUBLE（即使 `other` 和 `out` 完全合法）。
- **测试方案**: 构造 `self` 为 DT_DOUBLE、`other` 为 DT_DOUBLE 的合法输入调用 `aclnnMulGetWorkspaceSize`，验证返回值。再构造 `self` 为 DT_FLOAT、`other` 为 DT_DOUBLE 的情况，验证是否可以绕过该检查——体现了不一致性。

---

### Bug 2: mixDataType 路径缺少 IsMulSupportNonContiguous 检查

- **位置**: 第 488-490 行
- **类型**: 逻辑缺陷 / 缺少保护性检查
- **严重程度**: 高
- **描述**: 在 `aclnnMulGetWorkspaceSize` 中，当 `isMixDataType` 为 true 且 `isSupportNonContiguous`（即 `IsRegBase()`）为 true 时，直接将非连续 tensor（`selfWithStride`, `otherWithStride`）传入 `l0op::Mul`，而没有像第 505 行那样调用 `l0op::IsMulSupportNonContiguous(self, other)` 验证 kernel 是否真正支持该特定 tensor 的非连续访问。`IsRegBase()` 仅表示平台级别的能力，并不代表对具体 tensor shape/stride 组合的支持。
- **触发条件**: 在 910B 平台上，`self` 为 FP16/BF16、`other` 为 FP32（混合类型），且其中一个 tensor 为非连续存储且 stride 不被 Mul kernel 支持。
- **测试方案**: 构造 FP16 的非连续 tensor（如 transpose 后未 contiguous）与 FP32 tensor 做乘法，在 910B 平台运行，验证计算结果正确性或是否报错。

---

### Bug 3: CheckInplaceMulShape 缺少维度上限检查 (OP_CHECK_MAX_DIM)

- **位置**: 第 301-305 行
- **类型**: 输入校验缺失
- **严重程度**: 中
- **描述**: `CheckMulShape`（第 292 行）对 `self` 和 `other` 都执行了 `OP_CHECK_MAX_DIM(tensor, MAX_SUPPORT_DIMS_NUMS, return false)` 检查，但 `CheckInplaceMulShape` 没有执行相同检查。这使得 `aclnnInplaceMul` 路径可能接受维度数超过硬件支持上限的 tensor，导致后续 kernel 执行异常或越界。
- **触发条件**: 调用 `aclnnInplaceMulGetWorkspaceSize` 时传入维度数超过 `MAX_SUPPORT_DIMS_NUMS` 的 tensor。
- **测试方案**: 构造维度数为 MAX_SUPPORT_DIMS_NUMS+1（如 9 维）的 tensor，分别调用 `aclnnMulGetWorkspaceSize` 和 `aclnnInplaceMulGetWorkspaceSize`，验证前者返回错误而后者未正确拦截。

---

### Bug 4: aclnnInplaceMulGetWorkspaceSize 中 mixDataType 路径在非 RegBase 模式缺少非连续处理

- **位置**: 第 637-641 行
- **类型**: 逻辑缺陷
- **严重程度**: 低
- **描述**: 在 `aclnnInplaceMulGetWorkspaceSize` 中，`isMixDataType` 检查位于 Contiguous 转换之后（第 629-634 行已做 Contiguous），所以此处实际上 tensor 已经是连续的，问题不大。但代码结构上 `if (IsRegBase() && isMixDataType)` 这个分支在非 RegBase 模式下会将 mix-dtype 输入落入 else 分支做 Cast+Mul。这与 `aclnnMulGetWorkspaceSize` 中 mix-dtype 不做 Cast 的策略不一致。在非 RegBase 平台上，如果 kernel 实际支持 mix-dtype 输入，多余的 Cast 会导致性能损失和不必要的内存分配。
- **触发条件**: 在非 RegBase（如 910A）平台上，对 FP16+FP32 混合类型 tensor 调用 `aclnnInplaceMul`。
- **测试方案**: 在 910A 平台上执行 FP16 tensor inplace 乘 FP32 tensor，对比有无 Cast 时的性能和结果一致性。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 466 行 | 逻辑错误 | 高 | DT_DOUBLE 在支持列表中却被硬编码拒绝，且仅检查 self 不检查 other |
| 2 | 第 488-490 行 | 逻辑缺陷 | 高 | mixDataType 非连续路径缺少 IsMulSupportNonContiguous 保护检查 |
| 3 | 第 301-305 行 | 校验缺失 | 中 | CheckInplaceMulShape 缺少 OP_CHECK_MAX_DIM 维度上限检查 |
| 4 | 第 637-641 行 | 逻辑缺陷 | 低 | Inplace 路径在非 RegBase 平台对 mixDataType 不必要地执行 Cast |
