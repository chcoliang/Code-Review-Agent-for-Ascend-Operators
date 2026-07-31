**Bug:** 第59行 `(void)out;` 删除了空指针检查 `CheckNotNull2Tensor(self, out)`，当 self 或 out 为 nullptr 时，后续代码（如第96行 `self->IsEmpty()`）将触发空指针解引用崩溃。

**触发输入:** 传入 self=nullptr 或 out=nullptr 调用 aclnnSwishGetWorkspaceSize。
