# A218 代码审查报告 - swish_dag.h

## Bug 1: SwishNegOne 的 MemCfg 使用了不一致的 MemLevel::LEVEL_1

- **位置**: 第 153 行
- **类型**: 内存配置错误
- **严重程度**: 中
- **描述**: `SwishNegOne` 模板的 `MemCfg` 设置为 `MemOptCfg<MemLevel::LEVEL_1>`，而同文件中 `SwishZero`（第 172 行）和 `SwishOther`（第 189 行）均使用 `MemOptCfg<MemLevel::LEVEL_2>`。在 arch35 架构上，LEVEL_2 对应 UB（Unified Buffer）级别的内存优化，是 elementwise 算子的标准配置。使用 LEVEL_1 会导致 SwishNegOne 路径的 tiling 策略与其他路径不一致，可能造成 buffer 分配不足或性能退化，甚至在某些 shape 下触发内存越界。
- **触发条件**: 输入数据类型为 float16/bfloat16/float，且 scale 属性值为 -1.0 时，走入 SwishNegOne DAG 路径。
- **修复建议**: 将第 153 行的 `MemLevel::LEVEL_1` 改为 `MemLevel::LEVEL_2`：
  ```cpp
  using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
  ```
- **测试方案**:
  1. 使用 scale=-1.0，构造不同 shape（小 shape 如 [32]，大 shape 如 [1024, 1024]）的输入 tensor。
  2. 对比 SwishNegOne 路径的 tiling 参数与 SwishOther 路径在相同 shape 下的 buffer 分配策略。
  3. 验证大 shape 场景下是否出现 UB 内存越界或计算结果错误。
  4. 进行性能测试，对比修复前后 scale=-1 场景的执行耗时。

---

## 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 153 | 内存配置错误 | 中 | SwishNegOne 使用 LEVEL_1 与其他模板的 LEVEL_2 不一致 |
