**Bug:** 第62行 `CheckDtypeValidActivation(self, self, supportList)` 第二个参数应为 `out` 却误写为 `self`，导致输出张量的 dtype 从未被校验，非法的输出数据类型可绕过检查进入计算流程，造成结果错误或内存越界。

**触发输入:** 传入 self dtype=FLOAT16（合法），out dtype=INT64（非法）的张量调用 aclnnSwish。
