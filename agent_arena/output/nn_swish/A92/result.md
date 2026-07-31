**Bug:** 第52-55行 `CheckDim` 函数缺少对 `self` 的维度检查（缺少 `OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false)`），仅检查了 out。当 self 维度超过 MAX_SUPPORT_DIMS_NUMS 时无法拦截，后续 reshape 等操作可能访问越界或产生错误结果。

**触发输入:** 传入维度数超过 MAX_SUPPORT_DIMS_NUMS（如9维张量）的 self 输入。
