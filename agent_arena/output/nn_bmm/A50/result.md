# A50 BatchMatMul Code Review

## Bug: CheckParamsV2 空指针检查失败时返回错误的错误码 ACLNN_SUCCESS

- **位置**: 第 273 行, `CheckParamsV2` 函数
- **描述**: `CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_SUCCESS);` 当空指针检查失败时（`CheckNotNull` 返回 `false`），函数返回 `ACLNN_SUCCESS` 而非 `ACLNN_ERR_PARAM_NULLPTR`。这意味着即使传入空指针，`CheckParamsV2` 仍返回成功，调用方 `aclnnBatchMatMulGetWorkspaceSize` 中的 `CHECK_RET(ret == ACLNN_SUCCESS, ret)` 判断通过，后续代码对空指针解引用导致崩溃。
- **触发输入**:
  ```cpp
  // self: nullptr
  // mat2: valid tensor [2, 4, 5], DT_FLOAT16
  // out:  valid tensor [2, 3, 5], DT_FLOAT16
  // cubeMathType: 0
  aclnnBatchMatMulGetWorkspaceSize(nullptr, mat2, out, 0, &workspaceSize, &executor);
  ```
- **预期异常**: 应返回 `ACLNN_ERR_PARAM_NULLPTR`。实际行为：`CheckParamsV2` 返回 `ACLNN_SUCCESS`，后续 `CheckBmmResIsEmpty(self, mat2)` 对 `self`（nullptr）调用 `GetViewShape()` 导致 segfault。
