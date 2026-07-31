**Bug:** 第114行 `l0op::Swish(reshapeSelf, 1.0f, uniqueExecutor.get())` 使用硬编码 1.0f 而非变量 `scale`，导致用户通过 betaOptional 传入的自定义 scale 值被完全忽略，Swish 始终以 beta=1 执行。

**触发输入:** 传入 betaOptional=2.0 调用 aclnnSwish，期望 x*sigmoid(2*x) 但实际计算为 x*sigmoid(x)。
