**Bug:** 第109行 `float scale = 0.5f;` 默认 scale 应为 1.0f（标准 Swish 定义为 x*sigmoid(x)，即 beta=1）。当 betaOptional 为 nullptr 时，实际计算变为 x*sigmoid(0.5*x)，输出结果与标准 Swish 不一致。

**触发输入:** 不传入 betaOptional（betaOptional=nullptr）调用 aclnnSwish，期望标准 Swish 结果。
