# Ascend NPU 算子代码审查报告 - aclnn_gelu.cpp (A33)

## Bug 列表

### Bug 1: `workspaceSize` 和 `executor` 指针参数未做空指针校验

- **位置**: 第 83-84 行，`aclnnGeluGetWorkspaceSize` 函数入参
- **类型**: 空指针解引用 (Null Pointer Dereference)
- **严重程度**: 高
- **描述**: 函数参数 `workspaceSize`（第 97/115 行解引用）和 `executor`（第 98/116 行通过 `ReleaseTo` 使用）均为原始指针，但在使用前未进行空指针校验。若调用方传入 `nullptr`，将导致未定义行为（段错误/崩溃）。
- **触发条件**: 调用 `aclnnGeluGetWorkspaceSize(self, out, nullptr, &executor)` 或 `aclnnGeluGetWorkspaceSize(self, out, &ws, nullptr)`。
- **测试方案**: 构造合法的 `self` 和 `out` tensor，分别将 `workspaceSize` 和 `executor` 设为 `nullptr` 调用该函数，预期返回错误码而非崩溃。

---

### Bug 2: CheckShape 未校验输出 tensor 的维度上限

- **位置**: 第 51-55 行，`CheckShape` 函数
- **类型**: 校验遗漏 (Incomplete Validation)
- **严重程度**: 中
- **描述**: 函数仅对输入 `self` 做了 `OP_CHECK_MAX_DIM` 最大维度检查，对输出 `out` 使用 `(void)out` 直接忽略。若 `out` 的维度超过 `MAX_SUPPORT_DIMS_NUMS`，后续 kernel 计算可能越界或报错。
- **触发条件**: 传入 `self` 维度合法（如 4 维），但 `out` 维度超过 `MAX_SUPPORT_DIMS_NUMS`（如通过 view/reshape 得到高维输出 tensor）。
- **测试方案**: 构造 `self` 为 4D tensor，`out` 为超过最大支持维度的 tensor，调用接口验证是否正确报错。

---

### Bug 3: CheckShape 未校验输入输出 shape 一致性

- **位置**: 第 51-55 行，`CheckShape` 函数
- **类型**: 校验遗漏 (Incomplete Validation)
- **严重程度**: 高
- **描述**: GELU 是逐元素（element-wise）算子，要求输入 `self` 和输出 `out` 的 shape 完全一致。当前代码未校验两者 shape 是否匹配。若 shape 不同，`ViewCopy` 阶段可能导致内存越界写入或数据错误。
- **触发条件**: 传入 `self` shape 为 `[2, 3]`，`out` shape 为 `[3, 2]` 或 `[6]`，shape 不匹配但维度数可能相同。
- **测试方案**: 构造 shape 不同的 `self` 和 `out` tensor 调用接口，验证是否返回 shape 不匹配错误。

---

### Bug 4: 空 tensor 处理时未校验输出 tensor 状态

- **位置**: 第 96-100 行，空 tensor 分支
- **类型**: 逻辑缺陷 (Logic Error)
- **严重程度**: 中
- **描述**: 当 `self->IsEmpty()` 为 true 时直接返回成功，但未检查 `out` 是否也为空 tensor。若 `self` 为空但 `out` 非空（含有未初始化数据），调用方可能误认为 `out` 已被正确填充，导致后续使用脏数据。
- **触发条件**: `self` 为 shape `[0, 3]` 的空 tensor，`out` 为 shape `[2, 3]` 的非空 tensor（shape 校验缺失导致此情况可达）。
- **测试方案**: 构造空输入 tensor 和非空输出 tensor，调用接口后检查返回值和 `out` 内容是否符合预期。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第 83-84 行 | 空指针解引用 | 高 | `workspaceSize`/`executor` 未做空指针校验 |
| 2 | 第 51-55 行 | 校验遗漏 | 中 | 未校验 `out` 的维度上限 |
| 3 | 第 51-55 行 | 校验遗漏 | 高 | 未校验输入输出 shape 一致性 |
| 4 | 第 96-100 行 | 逻辑缺陷 | 中 | 空 tensor 时未校验 `out` 状态一致性 |
