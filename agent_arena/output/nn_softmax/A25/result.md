# aclnn_softmax.cpp 代码审查报告

## Bug 列表

### Bug 1: ASCEND910B 平台缺少 BF16 数据类型支持

- **位置**: 第 44-45 行，`ASCEND910B_DTYPE_SUPPORT_LIST` 定义
- **类型**: 功能缺陷
- **严重程度**: 高
- **描述**: 代码注释（第 40 行）明确说明 "AIC支持: DT_BF16, DT_FLOAT16, DT_FLOAT"，但 `ASCEND910B_DTYPE_SUPPORT_LIST` 中未包含 `DT_BF16`。Ascend 910B 及以上平台硬件支持 BF16，但数据类型白名单遗漏了该类型，导致 BF16 输入会被错误拒绝。
- **触发条件**: 在 Ascend 910B/C/D/E 平台上，输入 tensor 数据类型为 BF16 时，`CheckDtypeValid` 会返回 false，操作失败。
- **测试方案**: 在 910B 平台上构造 BF16 类型的输入 tensor 调用 `aclnnSoftmax`，验证是否能正常执行而非返回 `ACLNN_ERR_PARAM_INVALID`。

---

### Bug 2: 负数 dim 未归一化即传入底层算子

- **位置**: 第 133 行，`l0op::SoftmaxV2(self_contiguous, dim, uniqueExecutor.get())`
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `CheckDim` 函数（第 66-78 行）仅验证 dim 是否在合法范围 `[-selfDimNum, selfDimNum)` 内，但未将负数 dim 转换为正数索引（即 `dim += selfDimNum`）。负数 dim 直接传给 `SoftmaxV2` 底层算子，若底层算子不支持负数维度索引，将导致计算错误或崩溃。
- **触发条件**: 用户传入负数 dim（如 dim=-1 表示最后一维），底层 SoftmaxV2 算子不处理负数索引时触发。
- **测试方案**: 对一个 3 维 tensor 传入 dim=-1，对比与 dim=2 的计算结果是否一致；检查是否有异常或错误结果。

---

### Bug 3: 注释与实际调用不一致（SoftmaxGrad vs SoftmaxV2）

- **位置**: 第 132 行注释
- **类型**: 注释错误
- **严重程度**: 低
- **描述**: 注释写的是 "调用SoftmaxGrad算子kernel"，但实际调用的是 `l0op::SoftmaxV2`（前向 Softmax），并非梯度算子。这会误导后续维护者。
- **触发条件**: 代码维护/审查时产生误解。
- **测试方案**: 代码审查确认，将注释修正为 "调用SoftmaxV2算子kernel"。

---

### Bug 4: 输出 tensor 未检查维度上限

- **位置**: 第 80-85 行，`CheckShape` 函数
- **类型**: 校验遗漏
- **严重程度**: 中
- **描述**: `OP_CHECK_MAX_DIM` 仅对输入 `self` 进行了 8 维上限检查，未对输出 `out` 进行同样的检查。虽然后续 `OP_CHECK_SHAPE_NOT_EQUAL` 会校验 self 与 out 形状一致，但如果 out 的 view shape 与实际存储维度不同（如有额外的 size-1 维度被展开），可能绕过检查。
- **触发条件**: 输出 tensor 的 view shape 维度超过 8 维但与 self 的逻辑 shape 匹配时。
- **测试方案**: 构造维度超过 8 的输出 tensor，验证是否被正确拦截。

---

### Bug 5: 空 tensor 提前返回跳过了 dim 合法性校验

- **位置**: 第 92-95 行，`CheckParams` 函数
- **类型**: 校验顺序错误
- **严重程度**: 低
- **描述**: 当输入 tensor 为空时，函数在检查 dtype、dim、shape 之前就返回了 `ACLNN_SUCCESS`。这意味着即使用户传入非法的 dim 值（如 dim=100），对于空 tensor 也不会报错，与 PyTorch 行为不一致（PyTorch 对空 tensor 仍会校验 dim 范围）。
- **触发条件**: 输入为空 tensor 且 dim 值超出合法范围时，不会返回错误。
- **测试方案**: 构造空 tensor 传入非法 dim（如 dim=999），验证是否返回错误码。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第 44-45 行 | 功能缺陷 | 高 | ASCEND910B 平台 BF16 数据类型支持缺失 |
| 2 | 第 133 行 | 逻辑错误 | 高 | 负数 dim 未归一化直接传入底层算子 |
| 3 | 第 132 行 | 注释错误 | 低 | 注释误写为 SoftmaxGrad |
| 4 | 第 80-85 行 | 校验遗漏 | 中 | 输出 tensor 未检查 8 维上限 |
| 5 | 第 92-95 行 | 校验顺序 | 低 | 空 tensor 跳过 dim 合法性校验 |
