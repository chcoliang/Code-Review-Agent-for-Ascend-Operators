# GeLU 算子注册 代码审查报告

## Bug: 输入dtype注册了不支持的INT8类型，且输入输出dtype不匹配

- **位置**: `gelu_def.cpp` 第25行和第30行
  ```cpp
  .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT})   // Input x
  .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})   // Output y
  ```

- **描述**: 算子注册中，输入x的第一组dtype为 `ge::DT_INT8`，输出y的第一组dtype为 `ge::DT_BF16`。存在两个问题：
  1. GeLU算子涉及指数、除法等浮点运算，INT8类型不支持这些运算，Ascend 910B的Vector单元无法对INT8直接执行Exp/Div等指令。
  2. 输入输出第一组dtype不匹配（INT8 -> BF16），但GeLU是element-wise算子，正常情况下输入输出dtype应一致（应为BF16 -> BF16）。该错误配对会导致框架选中该dtype组合时，kernel内部Cast或计算逻辑无法正确处理INT8输入。

- **触发输入**: 当用户传入dtype为INT8的tensor作为GeLU算子的输入时，框架会匹配到第一组dtype配置（INT8->BF16）。

- **预期异常**: 编译期可能因kernel不支持INT8模板实例化而报错；若绕过编译，运行时AICore会因对INT8数据执行非法浮点指令而产生计算错误或硬件异常。
