**Bug**: 第170-171行 `CheckShapeValid` 中删除了对 `self` 的最大维度检查（`OP_CHECK_MAX_DIM(self, ...)`），仅保留了对 `mat2` 的检查，导致self维度超过6时未被拦截，引发后续越界或未定义行为。触发条件：self张量维度数为7或更多。
