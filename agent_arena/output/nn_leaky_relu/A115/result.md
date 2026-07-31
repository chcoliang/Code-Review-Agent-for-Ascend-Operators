**Bug**: 第32行 `DynamicCompileStaticFlag(false)` 应为 `true`。关闭动态编译静态化标志会导致算子无法利用 dynamic-compile-static 优化路径，在固定 shape 场景下可能导致性能退化或编译调度失败。

**触发输入**: 使用固定 shape 的 tensor 调用该算子时，无法走静态编译优化路径，性能下降或运行时报错。
