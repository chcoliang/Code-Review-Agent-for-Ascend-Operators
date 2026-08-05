# Ascend NPU 算子代码审查报告 - aclnn_softmax.cpp (A28)

## Bug 列表

### Bug 1: AXIS_LIMIT 定义但未使用，缺少维度上限校验

- **位置**: 第47行定义 `AXIS_LIMIT = 8`，第66-78行 `CheckDim` 函数
- **类型**: 逻辑缺陷 / 校验遗漏
- **严重程度**: 高
- **描述**: 代码定义了 `AXIS_LIMIT = 8` 并注释说明"底层算子不支持超过8维"，但 `CheckDim` 函数仅校验 dim 是否在tensor维度范围内，从未校验tensor的维度数是否超过8。当输入tensor维度超过8时，底层 SoftmaxV2 算子将产生未定义行为或运行失败。
- **触发条件**: 输入一个维度数大于8的tensor（如9维tensor）调用 aclnnSoftmax。
- **测试方案**: 构造一个9维tensor（如shape为[2,2,2,2,2,2,2,2,2]），调用 aclnnSoftmaxGetWorkspaceSize，预期应返回错误码而非继续执行。

---

### Bug 2: 负数 dim 未归一化即传入底层算子

- **位置**: 第131行 `l0op::SoftmaxV2(self_contiguous, dim, uniqueExecutor.get())`
- **类型**: 逻辑缺陷
- **严重程度**: 高
- **描述**: `CheckDim` 允许负数 dim（如 -1 表示最后一维），但在调用 `SoftmaxV2` 前未将负数 dim 转换为对应的正数索引。若底层 SoftmaxV2 算子不支持负数 dim 参数，将导致计算错误或崩溃。PyTorch 语义中 dim=-1 是合法的，需要在调用前归一化为 `dim + ndim`。
- **触发条件**: 调用 `aclnnSoftmax` 时传入负数 dim，如 dim=-1。
- **测试方案**: 对shape为[2,3,4]的tensor调用 aclnnSoftmax 并传入 dim=-1，验证结果是否与 dim=2 一致；若底层不支持负值，应观察到计算错误。

---

### Bug 3: 缺少对 workspaceSize 和 executor 指针的空指针校验

- **位置**: 第106-107行函数签名，第143-144行解引用
- **类型**: 空指针解引用风险
- **严重程度**: 中
- **描述**: `aclnnSoftmaxGetWorkspaceSize` 接收 `uint64_t* workspaceSize` 和 `aclOpExecutor** executor` 参数，但函数内未对这两个指针进行空指针校验即直接解引用（第121、122、143、144行）。若调用者传入空指针，将导致段错误。
- **触发条件**: 调用者传入 workspaceSize=nullptr 或 executor=nullptr。
- **测试方案**: 分别传入 workspaceSize=nullptr 和 executor=nullptr 调用该函数，预期应返回错误码 ACLNN_ERR_PARAM_NULLPTR 而非崩溃。

---

### Bug 4: 缺少输入数据类型提升（FP16/BF16 精度损失）

- **位置**: 第127-131行
- **类型**: 精度缺陷
- **严重程度**: 中
- **描述**: Softmax 运算涉及 exp 和求和操作，对 FP16/BF16 输入直接计算容易产生上溢/下溢。标准实现应在计算前将 FP16/BF16 提升为 FP32 进行计算，再将结果 Cast 回目标类型。当前代码仅在输出端做了 Cast（第135行），但未在输入端对低精度类型做 Cast 提升，可能导致数值精度问题。
- **触发条件**: 输入 FP16 或 BF16 类型的tensor，且数值范围较大（如包含绝对值>10的元素）。
- **测试方案**: 使用 FP16 tensor（含有较大值如65504）调用 aclnnSoftmax，对比与 FP32 计算结果的精度差异，检查是否存在 NaN/Inf。

---

### Bug 5: 注释错误 - "SoftmaxGrad" 应为 "Softmax/SoftmaxV2"

- **位置**: 第130行
- **类型**: 注释错误
- **严重程度**: 低
- **描述**: 注释写为"调用SoftmaxGrad算子kernel"，但实际调用的是前向算子 `SoftmaxV2`，而非反向梯度算子 SoftmaxGrad。这会误导代码维护者。
- **触发条件**: 代码审查 / 维护时产生误解。
- **测试方案**: 人工审查修正注释为"调用SoftmaxV2算子kernel"。

---

## 汇总表

| 编号 | 位置 | Bug类型 | 严重程度 | 简要描述 |
|------|------|---------|----------|----------|
| 1 | 第47行/第66-78行 | 逻辑缺陷/校验遗漏 | 高 | AXIS_LIMIT=8定义但未实际校验，超8维输入将导致底层算子失败 |
| 2 | 第131行 | 逻辑缺陷 | 高 | 负数dim未归一化为正数即传入SoftmaxV2底层算子 |
| 3 | 第106-107行/第143-144行 | 空指针解引用 | 中 | workspaceSize和executor指针未做空指针校验 |
| 4 | 第127-131行 | 精度缺陷 | 中 | FP16/BF16输入未提升至FP32计算，存在精度损失风险 |
| 5 | 第130行 | 注释错误 | 低 | 注释误写为"SoftmaxGrad"，实际调用SoftmaxV2前向算子 |
