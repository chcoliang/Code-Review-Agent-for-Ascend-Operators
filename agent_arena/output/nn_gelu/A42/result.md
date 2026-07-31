# GeLU 算子注册 AICore配置 代码审查报告

## Bug: DynamicCompileStaticFlag 设置为 false

- **位置**: `gelu_def.cpp` 第35行
  ```cpp
  aicoreConfig.DynamicCompileStaticFlag(false)
  ```

- **描述**: `DynamicCompileStaticFlag` 设置为 `false`，表示不启用动态编译静态化优化。在Ascend 910B（ascend910_95）平台上，对于支持动态shape的算子（`DynamicShapeSupportFlag(true)` 和 `DynamicRankSupportFlag(true)` 已开启），`DynamicCompileStaticFlag` 应设置为 `true` 以使能动态shape场景下的静态编译优化，提升执行性能。设置为 `false` 会导致动态shape场景下无法利用静态编译优化路径，每次shape变化都需要重新编译kernel，严重影响性能，且可能导致在某些固定shape调用场景下选不到最优tiling策略。

- **触发输入**: 任意动态shape输入，例如第一次调用shape为 `[1, 128]`，第二次调用shape为 `[1, 256]`，dtype为float16。

- **预期异常**: 不会产生功能性错误，但会导致严重的性能退化。在动态shape场景下，每次shape变化都会触发重新编译，编译开销显著增加，端到端推理延迟大幅上升。
