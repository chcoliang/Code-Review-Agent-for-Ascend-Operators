**Bug**: 第40行 `CheckNotNull` 函数中未对 `out` 参数进行空指针检查（`(void)out;` 直接忽略），导致后续第108行 `out->GetDataType()` 等调用在 `out` 为 nullptr 时触发空指针解引用崩溃。

**触发输入**: 调用 `aclnnLeakyReluGetWorkspaceSize` 时传入 `out = nullptr`。
