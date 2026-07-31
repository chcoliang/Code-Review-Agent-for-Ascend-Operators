# Code Review Result - A149

## Bug 1: scale 参数被硬编码为 1.0，用户传入值被忽略

- **位置**: 第 136 行
- **类型**: 逻辑错误 / 参数丢失
- **严重程度**: 高
- **描述**: 在 `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 函数中，调用内部函数 `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 时，第三个参数 `scale` 被硬编码为 `1.0`，而非使用用户传入的 `scale` 参数。这导致无论用户传入什么缩放因子，实际计算始终使用 scale=1.0，softmax 结果将完全错误。
- **触发条件**: 用户传入任何非 1.0 的 scale 值（例如 `scale = 0.125`，即 `1/sqrt(d_k)`），计算结果不会反映该缩放因子。
- **修复建议**: 将第 136 行的 `1.0` 替换为 `scale`：
  ```cpp
  auto result = aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize(x, mask, scale, false, y, workspaceSize, executor);
  ```
- **测试方案**:
  1. 构造输入 tensor x (shape=[1,1,4,4])，mask (shape=[1,1,4,4])，设置 scale=2.0。
  2. 调用 aclnnScaledMaskedSoftmaxGetWorkspaceSize 并执行计算。
  3. 对比输出 y 与手动计算 softmax(x * 2.0, mask) 的结果。
  4. 若 bug 存在，输出将等同于 softmax(x * 1.0, mask)，与预期不符。

---

## 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 136 | 逻辑错误 | 高 | scale 参数被硬编码为 1.0，用户传入值被忽略 |
