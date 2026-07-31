**Bug**: 第37-40行 `CheckNotNull` 函数缺少对 `negativeSlope` 参数的空指针检查，仅检查了 `self` 和 `out`。当 `negativeSlope` 为 nullptr 时，第103行 `negativeSlope->ToFloat()` 将触发空指针解引用崩溃。

**触发输入**: 调用 `aclnnLeakyReluGetWorkspaceSize` 时传入 `negativeSlope = nullptr`。
