**Bug:** 第40行 `ASCEND910B_DTYPE_SUPPORT_LIST` 错误地包含了 `DT_INT32`，Swish 是浮点激活函数，对整型数据执行 sigmoid 运算会产生错误结果或在 kernel 内部触发未定义行为。

**触发输入:** 在 Ascend 910B 上传入 dtype=INT32 的输入和输出张量调用 aclnnSwish。
