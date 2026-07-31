# A62 审查结果: opFile配置错误

## Bug

| 项目 | 内容 |
|------|------|
| **文件** | `batch_mat_mul_v3_def.cpp` |
| **位置** | 第121行, ascend910_95/ascend910_55 平台的 AICore 配置 |
| **问题代码** | `aicConfig.ExtendCfgInfo("opFile.value", "batch_mat_mul_v3_opt");` |
| **描述** | `opFile.value` 被错误设置为 `"batch_mat_mul_v3_opt"`，正确的值应为 `"batch_mat_mul_v3_apt"`。`opFile.value` 指定了该平台使用的kernel二进制文件名，`"batch_mat_mul_v3_opt"` 对应的kernel文件不存在或是错误的实现。对比第149行 mc62cm12a 平台正确使用了 `"batch_mat_mul_v3_apt"`，以及其他正确版本中 ascend910_95 同样使用 `"batch_mat_mul_v3_apt"`，说明此处为笔误。 |
| **触发条件** | 在 ascend910_95 或 ascend910_55 平台上执行 BatchMatMulV3 算子时。 |
| **预期异常** | 算子编译/加载阶段，框架根据 `opFile.value` 查找对应的 kernel binary 文件 `batch_mat_mul_v3_opt.o` 时找不到该文件，抛出 "kernel file not found" 或 "op binary load failed" 错误，导致模型编译失败，算子无法在 ascend910_95/ascend910_55 平台正常执行。 |
