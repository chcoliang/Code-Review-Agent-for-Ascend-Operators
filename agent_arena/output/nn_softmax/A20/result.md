# Softmax op_api 层代码审查报告

**审查文件**: `aclnn_softmax.cpp`  
**目标平台**: Ascend 910B, CANN 8.5.0

---

## Bug #1: 输出 tensor shape 未校验与输入一致

- **位置**: `CheckShape()` 函数 (第 80-86 行)
- **类型**: 参数校验缺失
- **严重程度**: 高
- **描述**: 函数注释声明"检查输入shape与输出shape是否一致"，但实现中仅检查了 `self` 的维度上限，对 `out` 直接 `(void)out` 忽略。未校验 `out` 的 shape 是否与 `self` 相同，也未校验 `out` 的维度是否超过 `AXIS_LIMIT`。当 `out` 的 shape 与 `self` 不匹配时，后续 `ViewCopy` 可能导致越界写入或静默数据错误。
- **触发输入**: `self` shape = [2, 3, 4], `out` shape = [2, 3] 或 [2, 3, 5]
- **预期异常**: 应返回 `ACLNN_ERR_PARAM_INVALID`，实际会进入计算流程导致未定义行为

---

## Bug #2: 输出 tensor 数据类型未校验（同族对称性缺失）

- **位置**: `CheckDtypeValid()` 函数 (第 58-64 行) 与 `aclnnSoftmaxGetWorkspaceSize` (第 137 行)
- **类型**: 参数校验不完整 / 同族对称性
- **严重程度**: 中
- **描述**: `CheckDtypeValid` 对 `out` 进行了 dtype 支持范围检查，但在空 tensor 分支（第 93-95 行）中提前返回 `ACLNN_SUCCESS`，跳过了对 `out` dtype 的校验。若 `self` 为空但 `out` 的 dtype 是不支持的类型（如 INT32），代码不会报错，导致与非空 tensor 时行为不对称。
- **触发输入**: `self` = empty tensor (shape [0, 3], dtype=float32), `out` = tensor (shape [0, 3], dtype=int32)
- **预期异常**: 应返回 `ACLNN_ERR_PARAM_INVALID`，实际返回 `ACLNN_SUCCESS`

---

## Bug #3: 910B 平台数据类型支持列表缺少 BF16 对应的 DT_DOUBLE 精度问题（注释与代码不一致）

- **位置**: `ASCEND910_DTYPE_SUPPORT_LIST` (第 41-42 行)
- **类型**: 数据类型支持范围潜在错误
- **严重程度**: 低
- **描述**: 注释说明 "AIC支持:DT_BF16, DT_FLOAT16, DT_FLOAT, AICPU支持 DT_DOUBLE"，但 `ASCEND910_DTYPE_SUPPORT_LIST`（用于 910A 平台）包含了 `DT_DOUBLE` 却不包含 `DT_BF16`。910A 注释并未说明支持 BF16，这部分逻辑看起来是正确的。但两个列表都包含 `DT_DOUBLE`，而 AIC core 不支持 double 运算，当 AIC core 路径被选择时可能导致运行时错误。该问题取决于底层调度策略，属于潜在风险。
- **触发输入**: `self` = tensor (dtype=float64) 在无 AICPU 支持的纯 AIC 路径下执行
- **预期异常**: 应在 dtype 校验阶段拦截或底层调度至 AICPU，实际可能运行失败

---

## Bug #4: `dim` 归一化前直接传入底层算子

- **位置**: `aclnnSoftmaxGetWorkspaceSize` 第 133 行
- **类型**: 边界条件处理缺失
- **严重程度**: 中
- **描述**: `CheckDim` 允许负数 dim（例如 -1 表示最后一个维度），校验通过后直接将负值 `dim` 传入 `l0op::SoftmaxV2`。如果底层 `SoftmaxV2` 不处理负数 dim 归一化，将导致错误结果或崩溃。标准做法应在传入前执行 `dim = dim < 0 ? dim + ndim : dim` 的归一化。
- **触发输入**: `self` shape = [2, 3, 4], `dim` = -1
- **预期异常**: 应等价于 `dim=2` 的 softmax，若底层不处理负数 dim，可能返回错误结果或运行时异常

---

## Bug #5: 空 tensor 场景下未校验 `workspaceSize` 和 `executor` 指针

- **位置**: `aclnnSoftmaxGetWorkspaceSize` 第 121-126 行
- **类型**: 空指针解引用风险
- **严重程度**: 中
- **描述**: 当 `self` 为空 tensor 时，直接对 `*workspaceSize` 赋值和调用 `uniqueExecutor.ReleaseTo(executor)`，但未校验 `workspaceSize` 和 `executor` 是否为空指针。非空路径中同样存在此问题（第 145-146 行），但空 tensor 分支更早触发。
- **触发输入**: `self` = empty tensor, `workspaceSize` = nullptr 或 `executor` = nullptr
- **预期异常**: 应返回 `ACLNN_ERR_PARAM_NULLPTR`，实际会触发段错误 (segfault)

---

## Bug #6: `CheckNotNull` 中 `out` 参数缺少 const 修饰（同族对称性）

- **位置**: `CheckNotNull` 函数签名 (第 32 行)
- **类型**: 接口对称性 / const 正确性
- **严重程度**: 低
- **描述**: `self` 为 `const aclTensor*`，而 `out` 为 `aclTensor*`（非 const）。在仅做空指针检查的函数中，`out` 也应声明为 `const aclTensor*`。虽然不会导致运行时错误，但破坏了接口对称性，在 const 上下文中可能导致编译告警。
- **触发输入**: N/A（编译期问题）
- **预期异常**: 部分严格编译环境下产生 const-qualification 告警

---

## 总结

| 严重程度 | 数量 |
|---------|------|
| 高 | 1 |
| 中 | 3 |
| 低 | 2 |

最关键的问题是 **输出 shape 完全未校验**（Bug #1），可能导致内存越界；其次是 **负数 dim 未归一化** 和 **空 tensor 路径跳过输出校验** 的问题。
