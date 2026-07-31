在`aclnnMul`异步执行后未调用`aclrtSynchronizeStream(stream)`就直接用`aclrtMemcpy`读取结果（第63-69行），导致读取到未完成计算的脏数据。触发条件：任何aclnnMul调用后立即读取输出tensor。
