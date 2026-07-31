**Bug**: 第65行 `CheckNotNull` 函数中对 `out` 参数使用 `(void)out;` 跳过了空指针检查，导致后续流程中 `out` 为 nullptr 时发生段错误。触发条件：传入 `out=nullptr` 调用 `aclnnMatmulGetWorkspaceSize`。
