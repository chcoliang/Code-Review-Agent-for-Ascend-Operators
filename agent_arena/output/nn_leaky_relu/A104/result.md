**Bug**: 第69行 `CHECK_RET(CheckNotNull(...), ACLNN_SUCCESS)` 错误码为 `ACLNN_SUCCESS`，应为 `ACLNN_ERR_PARAM_NULLPTR`。当参数为空指针时，CheckNotNull 返回 false，但 CHECK_RET 返回的是 ACLNN_SUCCESS（即不报错继续执行），导致后续空指针解引用崩溃。

**触发输入**: 调用 `aclnnLeakyReluGetWorkspaceSize` 时传入 `self = nullptr`。
