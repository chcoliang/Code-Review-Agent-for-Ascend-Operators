# GeLU 算子注册 opFile 代码审查报告

## Bug: opFile.value 配置为错误的文件名 "gelu_opt"

- **位置**: `gelu_def.cpp` 第41行
  ```cpp
  .ExtendCfgInfo("opFile.value", "gelu_opt");
  ```

- **描述**: `opFile.value` 指定了算子kernel二进制文件的名称，这里配置为 `"gelu_opt"`，但GeLU算子的标准APT kernel文件名应为 `"gelu_apt"`。该命名错误会导致运行时框架在查找已编译的算子kernel二进制时找不到对应文件（因为实际编译产出的文件名为 `gelu_apt`），从而无法加载kernel执行。

- **触发输入**: 任意合法的GeLU算子调用，例如输入shape为 `[4, 1024]`，dtype为float16。

- **预期异常**: 运行时报错，提示找不到算子kernel文件 `"gelu_opt"`，算子执行失败，可能抛出类似 `"Op kernel bin file not found"` 或 `"GetBinFilePath failed"` 的错误。
