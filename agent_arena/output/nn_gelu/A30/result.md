# aclnn_gelu.cpp 代码审查报告

## Bug 列表

### Bug 1: GELU 支持数据类型列表中包含不合理的 DT_INT32

- **位置**: 第 23 行
- **类型**: 逻辑错误 / 数据类型支持错误
- **严重程度**: 高
- **描述**: `DTYPE_SUPPORT_LIST` 中包含 `DataType::DT_INT32`。GELU 是高斯误差线性单元激活函数，其数学定义为 `x * 0.5 * (1 + erf(x / sqrt(2)))`，涉及浮点除法、erf 函数等连续数学运算，输出为非整数浮点值。对整数类型输入执行 GELU 计算在数学上无意义，底层 Ascend NPU 的 Gelu 算子内核通常也不支持 INT32 类型，会导致计算结果错误或运行时失败。
- **触发条件**: 用户传入 `DT_INT32` 类型的输入张量调用 `aclnnGelu`，参数校验会通过，但底层 `l0op::Gelu` 计算时产生未定义行为或错误结果。
- **测试方案**: 构造一个 INT32 类型的输入张量，调用 `aclnnGeluGetWorkspaceSize` 和 `aclnnGelu`，观察是否返回错误码或产生错误的计算结果。对比移除 DT_INT32 后是否正确拒绝该类型输入。

---

### Bug 2: 未对输出参数指针 workspaceSize 和 executor 进行空指针校验

- **位置**: 第 84-85 行（函数签名）、第 98-99 行和第 116-117 行（解引用处）
- **类型**: 空指针解引用 / 防御性编程缺失
- **严重程度**: 高
- **描述**: `aclnnGeluGetWorkspaceSize` 函数接收 `uint64_t *workspaceSize` 和 `aclOpExecutor **executor` 两个输出指针参数，但函数内部从未对这两个指针进行空指针检查，直接在第 98 行 `*workspaceSize = 0`、第 99 行 `uniqueExecutor.ReleaseTo(executor)`、第 116-117 行再次解引用。若调用者误传 `nullptr`，将导致段错误（Segmentation Fault）程序崩溃。
- **触发条件**: 调用者传入 `workspaceSize = nullptr` 或 `executor = nullptr`。
- **测试方案**: 分别以 `nullptr` 作为 `workspaceSize` 和 `executor` 参数调用 `aclnnGeluGetWorkspaceSize`，验证是否触发崩溃。修复后应返回 `ACLNN_ERR_PARAM_NULLPTR` 错误码。

---

### Bug 3: SoC 版本范围检查使用硬编码上界，BF16 支持判断不具备前向兼容性

- **位置**: 第 27-28 行
- **类型**: 逻辑错误 / 可维护性缺陷
- **严重程度**: 中
- **描述**: `CheckSocVersionIsSupportBf16` 使用 `GetSocVersion() >= SocVersion::ASCEND910B && GetSocVersion() <= SocVersion::ASCEND910E` 进行范围检查。如果未来新增的 SoC 版本（如 ASCEND910F 或更高系列）同样支持 BF16，该函数会错误返回 `false`，导致合法的 BF16 输入被拒绝。此外，该函数调用了两次 `GetCurrentPlatformInfo().GetSocVersion()`，存在不必要的重复调用开销。
- **触发条件**: 在 ASCEND910E 之后发布的支持 BF16 的新 SoC 平台上运行，传入 BF16 类型张量。
- **测试方案**: 模拟或在新 SoC 平台（版本号大于 ASCEND910E）上执行 BF16 输入的 GELU 计算，验证是否被错误拒绝。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第 23 行 | 逻辑错误 | 高 | GELU 不应支持 DT_INT32 整数类型，会导致计算错误 |
| 2 | 第 84-85, 98-99, 116-117 行 | 空指针解引用 | 高 | 未校验 workspaceSize/executor 输出指针，可能段错误 |
| 3 | 第 27-28 行 | 逻辑错误 | 中 | BF16 支持的 SoC 版本硬编码上界，缺乏前向兼容性 |
