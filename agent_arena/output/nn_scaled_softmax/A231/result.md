# A231 代码审查报告

**文件**: `scaled_masked_softmax_v2.h`

---

## Bug 1: MASK_VAL 值错误导致 Masked Softmax 语义错误

- **位置**: 第 22 行
- **类型**: 数值/语义错误
- **严重程度**: 高
- **描述**: `constexpr float MASK_VAL = 0.0f;` 是错误的。在 Scaled Masked Softmax 中，被 mask 掉的位置应该在 softmax 之前被设置为一个极大的负数（如 -10000.0f 或 -inf），使得 softmax 后这些位置的输出趋近于 0。使用 `MASK_VAL = 0.0f` 意味着被 mask 的位置在 softmax 输入中为 0，经过 softmax 后会产生非零正值（exp(0)=1），这完全违背了 masked softmax 的语义——被遮蔽的位置应该不参与注意力计算（输出为 0）。
- **触发条件**: 任何使用 mask 的场景都会触发。被 mask 的位置 softmax 输出不为 0，导致注意力机制错误地关注了应被屏蔽的 token。
- **测试方案**: 
  1. 构造输入 x 为全 1，mask 中部分位置为 false（需遮蔽），验证输出中被遮蔽位置是否趋近于 0。
  2. 对比正确实现（MASK_VAL=-10000.0f）的输出，验证差异。
  3. 在 Transformer 推理中使用 causal mask，验证模型输出是否正确。

---

## 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 22 | 数值/语义错误 | 高 | MASK_VAL=0.0f 应为负大数（如-10000.0f），导致 masked 位置 softmax 输出非零 |
