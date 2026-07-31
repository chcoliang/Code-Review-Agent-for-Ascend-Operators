**Bug**: 第75行 ascend910b的AICore配置中 `DynamicCompileStaticFlag(false)` 应为 `true`，关闭动态编译静态化会导致910B上无法生成静态kernel，算子编译或运行失败。触发条件：在Ascend 910B上使用MatMulV3算子。
