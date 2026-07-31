# A216 代码审查报告 - swish_tiling_arch35.cpp

## Bug 1: DT_FLOAT16 场景下 SwishNegOne 使用了错误的模板类型 bfloat16_t

- **位置**: 第 132 行
- **类型**: 模板类型参数错误
- **严重程度**: 高
- **描述**: 当 `outputDtype == ge::DT_FLOAT16` 且 `attrWork == TPL_SCALE_NEG_ONE` 时，代码实例化了 `SwishDag::SwishNegOne<bfloat16_t>::OpDag`，而应该使用 `SwishDag::SwishNegOne<half>::OpDag`。这导致 float16 输入在 scale=-1 场景下被错误地按 bfloat16 类型进行 tiling 计算，产生数据解析错误和计算结果完全错误。
- **触发条件**: 输入数据类型为 float16，且 scale 属性值为 -1.0。
- **修复建议**: 将第 132 行的 `bfloat16_t` 改为 `half`：
  ```cpp
  baseTilingResult = elewiseBaseTiling.DoTiling32B<SwishDag::SwishNegOne<half>::OpDag>();
  ```
- **测试方案**:
  1. 构造 float16 输入 tensor，设置 scale=-1.0。
  2. 执行 Swish 算子，对比输出与标杆值（numpy: `x * sigmoid(-x)`）。
  3. 验证 tiling 参数中的数据类型大小与 float16 (2 bytes) 一致而非 bfloat16。

---

## 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 132 | 模板类型参数错误 | 高 | DT_FLOAT16+scale=-1 场景错误使用 bfloat16_t 模板 |
