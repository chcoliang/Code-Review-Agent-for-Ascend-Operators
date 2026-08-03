# Gelu Tiling Arch35 代码审查报告

## Bug 列表

### Bug 1: FP16 分支使用了错误的模板类型 (bfloat16_t)

- **位置**: `gelu_tiling_arch35.cpp` 第 94 行
- **类型**: 模板选择错误 / 类型映射错误
- **严重程度**: 严重 (Critical)
- **描述**: 当 `outputDtype == ge::DT_FLOAT16` 时，代码调用了 `elewiseBaseTiling.DoTiling<GeluDAG<bfloat16_t>::OpDag>(*tiling)`，使用了 `bfloat16_t` 模板参数。正确的做法应该使用 `half` (即 `float16_t` / `half_t`) 类型的 DAG 模板。当前代码导致 FP16 和 BF16 两个分支使用了完全相同的 DAG 模板实例化，这意味着:
  1. FP16 场景的 tiling 计算基于 BF16 的 DAG 结构，buffer 大小、数据对齐等参数可能错误；
  2. 如果 GeluDAG 对 half 和 bfloat16_t 有不同的计算图结构（如中间精度提升策略不同），会导致运行时计算错误或内存越界。
- **触发条件**: 输入 tensor 数据类型为 `float16` (DT_FLOAT16) 时必然触发。
- **修复建议**:
  ```cpp
  // 第 94 行应修改为:
  baseTilingResult = elewiseBaseTiling.DoTiling<GeluDAG<half>::OpDag>(*tiling);
  ```
- **测试方案**:
  1. 构造 FP16 输入 tensor，执行 Gelu 算子，对比 CPU golden 结果验证精度；
  2. 检查 FP16 场景下 tiling 输出的 buffer 参数是否与 half 类型的 DAG 一致；
  3. 使用较大 shape 输入触发多核切分，验证无内存越界。

---

### Bug 2: tiling key 中 dType 与实际 DoTiling 模板不匹配 (FP16 场景)

- **位置**: `gelu_tiling_arch35.cpp` 第 93-94 行 与 第 112 行
- **类型**: Tiling 参数不一致
- **严重程度**: 严重 (Critical)
- **描述**: 第 93 行设置 `dType = TPL_FP16`，第 112 行用 `GET_TPL_TILING_KEY(1, dType)` 生成 tilingKey 为 FP16 对应的 key。但实际 DoTiling 使用的是 `GeluDAG<bfloat16_t>::OpDag`（BF16 模板）。这导致 kernel 侧根据 FP16 tilingKey 选择 FP16 kernel 实例，但 tiling 数据是按 BF16 DAG 计算的，两者不匹配，可能导致:
  1. kernel 按 FP16 数据宽度读取，但 tiling 按 BF16 计算的循环次数/块大小不正确；
  2. 运行时 UB 内存使用量计算错误。
- **触发条件**: 输入 tensor 数据类型为 `float16` 时触发。
- **修复建议**: 修复 Bug 1 后此问题自动消除（使 DoTiling 模板与 dType 一致）。
- **测试方案**:
  1. dump FP16 场景的 tiling data，验证 blockDim、循环次数等参数与 half 类型 DAG 期望值一致；
  2. 对比 FP16 和 BF16 场景的 tiling 输出，确认两者不同（在 half 和 bfloat16_t DAG 结构不同的前提下）。

---

## 汇总表

| 编号 | 位置 (行号) | Bug 类型 | 严重程度 | 简要描述 |
|------|-------------|----------|----------|----------|
| 1 | 第 94 行 | 模板选择错误 | Critical | FP16 分支错误使用 `GeluDAG<bfloat16_t>` 模板，应为 `GeluDAG<half>` |
| 2 | 第 93-94, 112 行 | Tiling 参数不一致 | Critical | tilingKey 标记为 FP16 但 tiling 数据按 BF16 DAG 生成，kernel 与 tiling 不匹配 |

## 根因分析

两个 Bug 本质上是同一个根因：第 94 行的 copy-paste 错误。开发者在编写 FP16 分支时，从 BF16 分支（第 97 行）复制了 `DoTiling<GeluDAG<bfloat16_t>::OpDag>` 但忘记将模板参数改为 `half`。修复第 94 行即可同时解决两个问题。
