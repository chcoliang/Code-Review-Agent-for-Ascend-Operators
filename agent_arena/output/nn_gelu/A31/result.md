# Ascend NPU 算子代码审查报告 - aclnn_gelu.cpp (A31)

## Bug 列表

### Bug 1: 空指针检查失败时返回成功状态码

- **位置**: 第 70 行
- **类型**: 逻辑错误 / 错误处理缺陷
- **严重程度**: 高 (High)
- **描述**: `CHECK_RET(CheckNotNull(self, out), ACLNN_SUCCESS);` 当 `CheckNotNull` 返回 `false`（即检测到空指针）时，`CHECK_RET` 宏会将第二个参数作为错误返回值返回给调用者。此处第二个参数为 `ACLNN_SUCCESS`，意味着即使输入为空指针，函数也会返回成功状态码，导致调用方无法感知错误，后续可能触发空指针解引用崩溃。正确写法应为 `CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_INVALID);`。
- **触发条件**: 当调用 `aclnnGeluGetWorkspaceSize` 时传入 `self = nullptr` 或 `out = nullptr`。
- **测试方案**: 
  1. 构造测试用例，传入 `self = nullptr`，验证返回值是否为错误码（而非 `ACLNN_SUCCESS`）。
  2. 构造测试用例，传入 `out = nullptr`，验证返回值是否为错误码。
  3. 验证后续流程不会因空指针而崩溃。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第70行 | 逻辑错误/错误处理缺陷 | 高 | 空指针检查失败时错误地返回 `ACLNN_SUCCESS` 而非错误码，导致空指针异常无法被上层捕获 |
