**Bug**: 第55行 `MAX_SUPPORT_MATMUL_DIMS_NUMS = 3` 错误地将最大支持维度从6改为3，导致合法的4/5/6维BatchMatMul输入被拒绝。触发条件：任何4维及以上的batch matmul输入（如shape [2,3,4,5] × [2,3,5,6]）。
