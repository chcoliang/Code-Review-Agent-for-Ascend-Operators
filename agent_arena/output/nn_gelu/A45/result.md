# GeLU Kernel DAG Cast模式 代码审查报告

## Bug: 当输入类型为float时，Cast操作冗余执行且GeluCustom仅支持float路径

- **位置**: `gelu_dag.h` 第47行
  ```cpp
  if constexpr(std::is_same_v<T, float>) {
  ```

- **描述**: `GeluCustom` 结构体中的实际计算逻辑被 `if constexpr(std::is_same_v<T, float>)` 守卫，这意味着只有当模板参数 `T` 为 `float` 时才会执行GeLU计算。然而在 `GeluDAG` 模板中，`T` 默认就是 `float`，而 `U` 是外部传入的输入/输出类型（如half/bfloat16_t）。当 `U=half` 或 `U=bfloat16_t` 时，DAG流程为：CopyIn(U) -> Cast(U->T, CAST_MODE_NONE) -> GeluCustom(T=float) -> Cast(T->U, CAST_MODE_NONE) -> CopyOut(U)。

  问题在于：当 `U=float`（即输入本身就是float32）时，Cast<float, float, CAST_MODE_NONE> 会被实例化。虽然从float到float的Cast在 `CAST_MODE_NONE` 下可能被优化为nop，但这依赖于框架实现。更关键的问题是：**当 `U != float`（如half/bfloat16_t）时，从T(float)到U(half)的输出Cast使用 `CAST_MODE_NONE`（截断模式）而非 `CAST_MODE_RINT`（四舍五入模式）**，这会导致float32到float16/bfloat16的精度转换采用截断方式，丢失精度，不符合GeLU算子的精度要求。

- **触发输入**: 输入dtype为float16或bfloat16，例如shape为 `[1, 1024]`，dtype=float16。计算结果从float32 Cast回float16时使用截断而非四舍五入。

- **预期异常**: 不会产生运行时crash，但计算精度下降。对比标杆结果会出现精度超差（ULP误差增大），在精度敏感的下游任务中可能导致模型效果退化。
