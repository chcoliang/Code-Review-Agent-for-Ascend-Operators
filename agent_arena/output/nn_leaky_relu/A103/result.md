**Bug**: 第35行 `ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST` 错误地包含了 `DT_INT32`。LeakyReLU 是浮点激活函数，INT32 类型不应支持，传入 INT32 tensor 会通过 dtype 校验但在 kernel 浮点乘法计算中产生未定义行为或错误结果。

**触发输入**: 在 Ascend910B 上传入 dtype 为 INT32 的 tensor 调用 `aclnnLeakyReluGetWorkspaceSize`。
