**Bug:** 第35行 `const int64_t ASCEND_WORKSPACE = 16777216 * 2;` 实际值为 32MB，但注释标注为 `// 16 * 1024 * 1024`（16MB）。workspace 分配为预期的两倍，浪费 HBM 内存资源；若下游代码依赖 16MB 约定进行偏移计算则可能导致数据错位。

**触发输入:** 任何 Swish 算子调用都会分配 32MB workspace（预期 16MB），在多算子并行场景下可能因 HBM 不足导致 OOM。
