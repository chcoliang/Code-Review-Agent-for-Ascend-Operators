**Bug:** 第34行 `DynamicCompileStaticFlag(false)` 应为 `true`。关闭动态编译静态化标志后，动态 shape 场景无法利用静态编译优化，每次 shape 变化都需重新编译 kernel，导致严重性能下降甚至在不支持纯动态编译的场景下执行失败。

**触发输入:** 使用动态 shape 输入（如不同 batch size）连续调用 Swish 算子。
