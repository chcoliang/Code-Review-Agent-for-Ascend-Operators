**Bug**: 第35行 `ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST` 缺少 `DT_BF16` 类型支持。Ascend910B 本应支持 BF16，但支持列表与910相同只有 FLOAT/FLOAT16/DOUBLE，导致 BF16 输入被错误拒绝。

**触发输入**: 在 Ascend910B 上传入 dtype 为 BF16 的 tensor 调用 `aclnnLeakyReluGetWorkspaceSize`，将返回 `ACLNN_ERR_PARAM_INVALID`。
