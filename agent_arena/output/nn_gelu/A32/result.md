# Ascend NPU 算子代码审查报告 - aclnn_gelu.cpp (A32)

## Bug 列表

### Bug 1: `aclnnGeluGetWorkspaceSize` 未对输出参数 `workspaceSize` 进行空指针检查

- **位置**: 第 84-85 行（函数签名）及第 109 行（解引用处）
- **类型**: 空指针解引用
- **严重程度**: 高
- **描述**: 函数参数 `uint64_t *workspaceSize` 在第 109 行通过 `*workspaceSize = uniqueExecutor->GetWorkspaceSize()` 直接解引用赋值，但函数入口未对该指针进行空指针校验。若调用者传入 nullptr，将导致段错误（Segmentation Fault）。
- **触发条件**: 调用者传入 `workspaceSize = nullptr`。
- **测试方案**: 调用 `aclnnGeluGetWorkspaceSize(validSelf, validOut, nullptr, &executor)`，预期应返回错误码而非崩溃。

---

### Bug 2: `aclnnGeluGetWorkspaceSize` 未对输出参数 `executor` 进行空指针检查

- **位置**: 第 84-85 行（函数签名）及第 110 行（使用处）
- **类型**: 空指针解引用
- **严重程度**: 高
- **描述**: 函数参数 `aclOpExecutor **executor` 在第 110 行通过 `uniqueExecutor.ReleaseTo(executor)` 使用，但函数入口未检查 `executor` 是否为 nullptr。若 `ReleaseTo` 内部解引用该二级指针（`*executor = ...`），将导致段错误。
- **触发条件**: 调用者传入 `executor = nullptr`。
- **测试方案**: 调用 `aclnnGeluGetWorkspaceSize(validSelf, validOut, &ws, nullptr)`，预期应返回错误码而非崩溃。

---

### Bug 3: `aclnnGelu` 未对 `executor` 和 `stream` 参数进行空指针检查

- **位置**: 第 114 行（函数签名）及第 117 行（传递给 `CommonOpExecutorRun`）
- **类型**: 空指针解引用 / 参数校验缺失
- **严重程度**: 高
- **描述**: `aclnnGelu` 函数直接将 `workspace`、`executor`、`stream` 传递给 `CommonOpExecutorRun`，未在本层进行任何空指针校验。若 `executor` 为空，底层函数可能直接崩溃；若 `stream` 为空，则算子无法正确下发到 NPU 流上执行。
- **触发条件**: 调用者传入 `executor = nullptr` 或 `stream = nullptr`。
- **测试方案**: 分别传入空的 executor 和 stream，验证是否能安全返回错误码。

---

### Bug 4: BF16 支持的 SoC 版本范围检查使用闭区间可能导致误判

- **位置**: 第 27-28 行 `CheckSocVersionIsSupportBf16` 函数
- **类型**: 逻辑缺陷 / 兼容性风险
- **严重程度**: 中
- **描述**: 使用 `>= ASCEND910B && <= ASCEND910E` 的闭区间范围判断 BF16 支持。如果未来在 910B 和 910E 之间插入不支持 BF16 的 SoC 版本枚举值，或者 910E 之后的新型号支持 BF16 但不在此范围内，该函数会给出错误结果。建议使用白名单方式或平台能力查询接口。
- **触发条件**: 新增 SoC 版本枚举值落在 [910B, 910E] 区间内但不支持 BF16，或新 SoC（如 910F）支持 BF16 但枚举值 > 910E。
- **测试方案**: 模拟新 SoC 版本返回值，验证 BF16 数据类型是否被正确允许或拒绝。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 109 行 | 空指针解引用 | 高 | `workspaceSize` 指针未做空检查即解引用 |
| 2 | 第 110 行 | 空指针解引用 | 高 | `executor` 二级指针未做空检查即使用 |
| 3 | 第 114-117 行 | 参数校验缺失 | 高 | `aclnnGelu` 未校验 executor/stream 空指针 |
| 4 | 第 27-28 行 | 逻辑缺陷 | 中 | BF16 SoC 版本范围硬编码，扩展性差 |
