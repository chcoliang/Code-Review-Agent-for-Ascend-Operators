**Bug**: 第38行 `ExtendCfgInfo("opFile.value", "leaky_relu_opt")` 中 kernel 二进制文件名错误，应为 `"leaky_relu_apt"` 而非 `"leaky_relu_opt"`。运行时将找不到正确的 kernel 文件导致算子执行失败。

**触发输入**: 任意合法输入调用 LeakyRelu 算子，运行时因找不到 "leaky_relu_opt" 对应的 kernel 二进制而报错。
