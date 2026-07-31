**Bug:** 第39-40行 `ASCEND910B_DTYPE_SUPPORT_LIST` 缺少 `DT_BF16`，仅包含 `{DT_FLOAT, DT_FLOAT16}`，导致 Ascend 910B 上 BF16 类型的输入张量被错误拒绝，无法执行 Swish 计算。

**触发输入:** 在 Ascend 910B 上传入 dtype=BF16 的输入和输出张量。
