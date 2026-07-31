循环中`aclrtMalloc`分配workspace内存后从未调用`aclrtFree(workspaceAddr)`释放（第60-61行分配，循环结束无释放），导致设备内存泄漏，多次迭代后NPU显存耗尽。触发条件：循环多次调用aclnnMul且workspaceSize > 0。
