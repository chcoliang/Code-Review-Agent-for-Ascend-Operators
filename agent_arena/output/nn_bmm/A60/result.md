# A60 审查结果: dtype注册对齐问题

## Bug

| 项目 | 内容 |
|------|------|
| **文件** | `batch_mat_mul_v3_def.cpp` |
| **位置** | 第24行, 全局算子定义 `this->Input("x1")` 的 DataType 列表 |
| **问题代码** | `.DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16})` |
| **描述** | x1的第一个dtype注册为 `DT_INT8`，但对应的x2第一个dtype为 `DT_FLOAT16`，输出y第一个dtype为 `DT_FLOAT16`。BatchMatMulV3全局定义中 INT8 x FP16 不是合法的matmul组合。对比ascend910b/ascend910_93的AICore配置（第59-67行），该平台配置并未包含INT8输入的dtype组合，全局定义的第一列dtype组合与平台AICore配置不对齐。正确的x1第一个dtype应为 `DT_FLOAT16`（与x2对齐，形成 FP16 x FP16 -> FP16 的合法组合）。 |
| **触发条件** | 当框架尝试通过全局dtype匹配选择第一组dtype组合时（x1=INT8, x2=FP16），由于ascend910b的AICore配置中没有注册对应的INT8输入组合，会导致算子选择失败或数据类型推导错误。 |
| **预期异常** | 算子注册阶段dtype校验失败，或在图编译阶段由于全局dtype与AICore平台配置不匹配导致 `OP_TILING_CHECK` 报错，输出 "data type not supported" 类错误；若侥幸匹配成功，INT8数据以FP16格式参与Cube计算将产生精度完全错误的结果。 |
