# A61 审查结果: AICore配置问题

## Bug

| 项目 | 内容 |
|------|------|
| **文件** | `batch_mat_mul_v3_def.cpp` |
| **位置** | 第60行, ascend910b/ascend910_93 的 AICore 配置 |
| **问题代码** | `aicConfig.DynamicCompileStaticFlag(false)` |
| **描述** | `DynamicCompileStaticFlag` 被错误设置为 `false`。对于 ascend910b 和 ascend910_93 平台，该标志应设为 `true`，表示动态shape算子编译后可缓存为静态tiling，避免每次执行时重新计算tiling。设为 `false` 意味着禁用了动态编译静态化优化，每次执行动态shape的BatchMatMulV3时都需要重新进行tiling计算，无法利用编译缓存。 |
| **触发条件** | 在 ascend910b 或 ascend910_93 平台上运行动态shape的BatchMatMulV3算子时，多次以相同shape执行该算子。 |
| **预期异常** | 功能不会出错，但性能严重劣化。由于 `DynamicCompileStaticFlag=false`，框架无法将已编译的动态tiling结果缓存为静态结果复用，导致每次推理都需要重新执行tiling计算流程，在高频调用场景下（如模型推理循环）产生显著的host侧开销，推理延迟增大。 |
