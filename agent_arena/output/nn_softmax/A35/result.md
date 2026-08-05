# Ascend NPU 算子代码审查报告 - aclnn_softmax.cpp (A35)

## Bug 列表

### Bug 1: 未对 workspaceSize 和 executor 指针进行空指针检查

- **位置**: 第 109 行 `aclnnSoftmaxGetWorkspaceSize` 函数入口
- **类型**: 空指针解引用
- **严重程度**: 高
- **描述**: 函数参数 `workspaceSize` 和 `executor` 为外部传入的指针，在第 123 行 `*workspaceSize = 0`、第 141 行 `*workspaceSize = uniqueExecutor->GetWorkspaceSize()` 和第 124/142 行 `uniqueExecutor.ReleaseTo(executor)` 处直接解引用，但函数入口未做空指针校验。若调用者传入 nullptr，将导致段错误（segfault）。
- **触发条件**: 调用者传入 `workspaceSize = nullptr` 或 `executor = nullptr`。
- **测试方案**: 调用 `aclnnSoftmaxGetWorkspaceSize(validSelf, 0, validOut, nullptr, &executor)` 和 `aclnnSoftmaxGetWorkspaceSize(validSelf, 0, validOut, &ws, nullptr)`，验证是否安全返回错误码而非崩溃。

---

### Bug 2: 负数 dim 未归一化即传入底层算子

- **位置**: 第 129 行 `l0op::SoftmaxV2(self, dim, uniqueExecutor.get())`
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `CheckDim` 函数（第 66-78 行）仅校验 `dim` 是否在 `[-selfDimNum, selfDimNum)` 范围内，但并未将负数 dim 转换为正数索引（即 `dim += selfDimNum`）。随后在第 129 行将原始的负数 dim 直接传给 `l0op::SoftmaxV2`。若底层 SoftmaxV2 算子不支持负数 axis，将导致计算错误或内部异常。
- **触发条件**: 调用时传入负数 dim，如 `dim = -1` 对一个 shape 为 [2,3,4] 的 tensor 执行 softmax。
- **测试方案**: 构造 shape=[2,3,4] 的 tensor，分别传入 `dim=-1`、`dim=-2`、`dim=-3`，验证输出结果是否与对应正数 dim (2,1,0) 结果一致。

---

### Bug 3: 空 tensor 提前返回绕过 dim 合法性校验

- **位置**: 第 93-95 行（`CheckParams` 函数内）
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: 当 `self->IsEmpty()` 为 true 时，函数直接返回 `ACLNN_SUCCESS`，跳过了后续的 `CheckDim`、`CheckDtypeValid` 和 `CheckShape` 检查。这意味着即使用户传入非法的 dim 值（如 dim=100）或非法的 dtype/shape，也不会报错，掩盖了调用者的编程错误，与 PyTorch 行为不一致（PyTorch 对空 tensor 仍然校验 dim）。
- **触发条件**: 传入一个空 tensor（某维度为 0），同时传入越界的 dim 值。
- **测试方案**: 构造 shape=[2,0,4] 的空 tensor，传入 `dim=10`，验证是否返回参数错误（当前错误地返回成功）。

---

### Bug 4: 注释与实际代码不符（Copy-Paste 错误）

- **位置**: 第 128 行注释 `// 调用SoftmaxGrad算子kernel`
- **类型**: 注释错误
- **严重程度**: 低
- **描述**: 注释写的是 "SoftmaxGrad"（反向算子），但实际调用的是 `l0op::SoftmaxV2`（前向算子）。这是明显的 copy-paste 错误，会误导代码维护者。
- **触发条件**: 代码审查/维护时产生误导。
- **测试方案**: 代码 review 即可发现；将注释修正为 "调用SoftmaxV2算子kernel"。

---

### Bug 5: CheckNotNull 和 CheckShape 中 out 参数 const 修饰不一致

- **位置**: 第 32 行 `CheckNotNull(const aclTensor *self, aclTensor *out)`、第 58 行 `CheckDtypeValid(const aclTensor *self, const aclTensor *out)`、第 80 行 `CheckShape(const aclTensor *self, aclTensor *out)`
- **类型**: 接口设计缺陷
- **严重程度**: 低
- **描述**: `out` 参数在不同检查函数中 const 修饰不一致。`CheckNotNull` 和 `CheckShape` 接收非 const 的 `aclTensor *out`，而 `CheckDtypeValid` 接收 `const aclTensor *out`。这些函数都只读取 out 的属性，不修改它，应统一使用 `const aclTensor *out`。不一致的接口可能导致后续维护者误认为函数会修改 out。
- **触发条件**: 编译期不会报错，但在代码重构或增加 const 约束时可能暴露问题。
- **测试方案**: 将所有检查函数的 `out` 参数统一为 `const aclTensor *out`，验证编译通过且功能正常。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 109 行函数入口 | 空指针解引用 | 高 | workspaceSize/executor 指针未做空指针检查 |
| 2 | 第 129 行 | 逻辑错误 | 高 | 负数 dim 未归一化为正数即传入 SoftmaxV2 |
| 3 | 第 93-95 行 | 逻辑错误 | 中 | 空 tensor 提前返回绕过 dim/dtype/shape 校验 |
| 4 | 第 128 行 | 注释错误 | 低 | 注释写 SoftmaxGrad 实为 SoftmaxV2 前向 |
| 5 | 第 32/80 行 | 接口设计缺陷 | 低 | out 参数 const 修饰不一致 |
