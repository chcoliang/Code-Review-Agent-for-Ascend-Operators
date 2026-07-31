**Bug:** 第40行 `ExtendCfgInfo("opFile.value", "swish_opt")` kernel 文件名错误，应为 `"swish_apt"`。运行时将无法找到正确的 kernel 二进制文件，导致算子执行失败。

**触发输入:** 任何输入调用 Swish 算子时，框架加载 kernel 阶段即会因找不到 "swish_opt" 对应的 kernel 文件而报错。
