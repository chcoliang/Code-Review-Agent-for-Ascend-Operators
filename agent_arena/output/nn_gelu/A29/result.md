# Ascend NPU 算子代码审查报告 - aclnn_gelu.cpp (A29)

## Bug 列表

### Bug 1: `out` 参数未进行空指针检查

- **位置**: 第 31-35 行，`CheckNotNull` 函数
- **类型**: 空指针解引用 (Null Pointer Dereference)
- **严重程度**: 严重 (Critical)
- **描述**: `CheckNotNull` 函数仅对 `self` 进行了空指针检查，而对 `out` 参数使用 `(void)out;` 显式忽略，未做任何空指针校验。然而在后续的 `CheckDtypeValid`（第46行 `out->GetDataType()`）、`CheckFormat`（第60行 `out->GetStorageFormat()`）和 `CheckShape`（第54行）中均直接解引用了 `out` 指针。若调用者传入 `out = nullptr`，将导致程序崩溃（段错误）。
- **触发条件**: 调用 `aclnnGeluGetWorkspaceSize` 时传入 `out = nullptr`。
- **测试方案**:
  ```cpp
  // 构造合法的 self tensor，但 out 传入 nullptr
  aclTensor *self = createValidTensor();
  aclTensor *out = nullptr;
  uint64_t workspaceSize;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnGeluGetWorkspaceSize(self, out, &workspaceSize, &executor);
  // 期望返回 ACLNN_ERR_PARAM_NULLPTR，而非崩溃
  EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
  ```
- **修复建议**: 在 `CheckNotNull` 中增加对 `out` 的空指针检查：
  ```cpp
  static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
  }
  ```

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第31-35行 `CheckNotNull` | 空指针解引用 | 严重 | `out` 参数未做空指针检查，后续函数直接解引用会导致崩溃 |
