# Code Review: aclnn_scaled_masked_softmax.cpp (A151)

## Bug 列表

### Bug 1: fixedTriuMask 参数校验逻辑反转

- **位置**: 第132-135行
- **类型**: 逻辑错误
- **严重程度**: 严重 (Critical)
- **描述**: 代码中 `if (!fixedTriuMask)` 判断为 false 时报错，错误信息却写 "the param fixedTriuMask only suppport false"。逻辑与错误信息自相矛盾。当 `fixedTriuMask=false` 时（即 `!fixedTriuMask` 为 true）会报错返回，但错误信息说"只支持 false"。结合第136行始终传 `false` 给内层函数来看，应当是 `fixedTriuMask=true` 时报错（即条件应为 `if (fixedTriuMask)`），或者错误信息应改为 "only support true"。根据函数名 "ScaledMaskedSoftmax" 和内层固定传 false 的行为，最可能的意图是：只支持 `fixedTriuMask=false`，当用户传入 `true` 时报错。因此条件应为 `if (fixedTriuMask)` 而非 `if (!fixedTriuMask)`。
- **触发条件**: 当用户传入 `fixedTriuMask=false`（本应合法的值）时会错误地返回失败；传入 `fixedTriuMask=true`（本应非法的值）时会错误地通过校验。
- **测试方案**: 分别以 `fixedTriuMask=true` 和 `fixedTriuMask=false` 调用 `aclnnScaledMaskedSoftmaxGetWorkspaceSize`，验证 false 时应成功，true 时应返回 `ACLNN_ERR_PARAM_INVALID`。

---

### Bug 2: 错误日志中 dim4 范围描述与实际限制不一致

- **位置**: 第106行
- **类型**: 日志/提示信息错误
- **严重程度**: 轻微 (Minor)
- **描述**: 错误信息固定写 "Expected x and mask dim4 in range of (0, 4096]"，但当平台为 ASCEND910_95 时，实际上限为 8192（`D_LIMIT_D`）。该日志会误导用户，使其不知道在该平台上上限已扩展为 8192。
- **触发条件**: 在 ASCEND910_95 平台上，`x` 的第4维大小为 5000（在 4096~8192 之间）时不会触发错误（因为实际限制为8192），但如果维度超过 8192 触发该错误时，日志信息会显示错误的范围上限 4096。
- **测试方案**: 在 ASCEND910_95 平台上传入 dim3=8193 的 tensor，检查错误信息应正确显示上限为 8192。

---

### Bug 3: 未校验 x 和 y 的 shape 一致性

- **位置**: 第113-123行 (`CheckParams` 函数)
- **类型**: 校验缺失
- **严重程度**: 中等 (Medium)
- **描述**: `CheckParams` 函数校验了 x 与 mask 的 shape 关系，以及 x 与 y 的 dtype 一致性，但未校验输出 tensor `y` 的 shape 是否与输入 `x` 的 shape 一致。Softmax 是逐元素操作，输出 shape 必须与输入相同。若 y 的 shape 与 x 不匹配，可能导致内存越界写入或计算结果错误。
- **触发条件**: 用户传入 shape 与 x 不同的 y tensor（例如 y 的某一维更小），则运行时可能产生未定义行为。
- **测试方案**: 传入 x shape 为 [2,4,8,64]、y shape 为 [2,4,8,32] 的 tensor，验证是否返回参数错误。

---

### Bug 4: 未校验 workspaceSize 和 executor 指针是否为空

- **位置**: 第126-138行 (`aclnnScaledMaskedSoftmaxGetWorkspaceSize` 函数)
- **类型**: 校验缺失
- **严重程度**: 中等 (Medium)
- **描述**: 函数参数 `workspaceSize` 和 `executor` 均为输出型指针参数，但未进行空指针校验。若调用者传入 nullptr，内层函数 `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 可能直接解引用导致段错误。
- **触发条件**: 调用者传入 `workspaceSize=nullptr` 或 `executor=nullptr`。
- **测试方案**: 分别传入 nullptr 的 workspaceSize 和 executor，验证应返回错误码而非崩溃。

---

### Bug 5: namespace 匿名空间内声明 extern 函数（与 extern "C" 交互问题）

- **位置**: 第25-27行 `extern "C"` 与第29行匿名 namespace
- **类型**: 链接/编译问题
- **严重程度**: 中等 (Medium)
- **描述**: `extern "C"` 块内部包含了匿名 namespace（第29-124行）。匿名 namespace 中的符号具有内部链接（internal linkage），但其中第39-44行又用 `extern` 声明了外部函数 `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 和 `aclnnInnerScaledMaskedSoftmaxV2`。在 `extern "C"` 块中、匿名 namespace 内的 extern 声明可能导致链接器行为不确定或编译警告，且语义混乱。这些 extern 声明应放在匿名 namespace 外部。
- **触发条件**: 在某些编译器/链接器版本下可能出现链接失败或符号不可见。
- **测试方案**: 使用不同版本编译器编译，检查是否有链接警告或错误。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第132-135行 | 逻辑错误 | 严重 | `fixedTriuMask` 校验条件取反，导致合法值被拒绝、非法值被放行 |
| 2 | 第106行 | 日志信息错误 | 轻微 | 错误信息中范围上限硬编码4096，未适配910_95平台的8192限制 |
| 3 | 第113-123行 | 校验缺失 | 中等 | 未校验输出 y 与输入 x 的 shape 一致性 |
| 4 | 第126-138行 | 校验缺失 | 中等 | 未校验 workspaceSize/executor 输出指针是否为空 |
| 5 | 第25-29行 | 链接问题 | 中等 | 匿名namespace内extern声明与extern "C"交互语义混乱 |
