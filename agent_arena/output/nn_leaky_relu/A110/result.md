**Bug**: 第61-63行 `CheckShape` 函数缺少 `OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false)` 维度上限检查。超过最大支持维度数的 tensor 不会被拦截，可能导致 kernel 内部 tiling 计算越界或缓冲区溢出。

**触发输入**: 传入维度数超过 MAX_SUPPORT_DIMS_NUMS（通常为8）的 tensor，例如 shape 为 [1,1,1,1,1,1,1,1,1] 的 9 维 tensor。
