# Gelu Tiling (arch35) 代码审查报告

## Bug 1: FP16 类型使用了错误的 DAG 模板

- **位置**: 第 94 行
- **类型**: 模板选择错误
- **严重程度**: 高
- **描述**: 当 `outputDtype == ge::DT_FLOAT16` 时，应使用 `GeluDAG<half>::OpDag` (即 float16 对应的模板)，但代码错误地使用了 `GeluDAG<bfloat16_t>::OpDag`。这会导致 FP16 数据被当作 BF16 处理，计算结果完全错误。
- **触发条件**: 输入/输出数据类型为 float16 时触发。
- **测试方案**: 构造 float16 类型的输入张量，调用 Gelu 算子，对比输出结果与标准 GELU 数学公式的参考值，验证精度是否在合理范围内。预期当前代码会产生明显的数值偏差。

## Bug 2: FP16 分支的 dType 枚举与模板不一致

- **位置**: 第 93-94 行
- **类型**: 类型映射不一致
- **严重程度**: 高
- **描述**: 第 93 行设置 `dType = TPL_FP16`，但第 94 行使用 `GeluDAG<bfloat16_t>::OpDag` 模板进行 tiling 计算。`TPL_FP16` 对应的 tiling key 与实际使用的 BF16 DAG 模板所期望的资源分配不匹配，可能导致 UB 空间计算错误、数据搬运异常或运行时崩溃。
- **触发条件**: 输入/输出数据类型为 float16 时触发。
- **测试方案**: 使用 float16 输入运行算子，检查 tiling 参数（block 划分、loop 次数等）是否与 float16 元素大小（2 字节）一致，而非与 BF16 模板计算的结果一致。验证是否出现内存越界或结果错误。

## Bug 3: FP16 应使用 half 类型模板而非 bfloat16_t

- **位置**: 第 94 行
- **类型**: 模板参数错误
- **严重程度**: 高
- **描述**: 正确的写法应为 `elewiseBaseTiling.DoTiling<GeluDAG<half>::OpDag>(*tiling)`。`half`（即 `float16_t`）是 Ascend C 中 FP16 的标准类型。当前使用 `bfloat16_t` 会使得 DAG 图中的计算节点按照 BF16 的精度特征（更大动态范围、更低精度）来配置流水线，而非 FP16 的特征（更高精度、较小动态范围），产生错误的计算图。
- **触发条件**: 输入/输出数据类型为 float16 时触发。
- **测试方案**: 分别用 FP16 和 BF16 输入运行算子，检查 FP16 路径生成的 tiling 参数是否与 BF16 路径完全相同（不应相同）。若相同则确认 bug 存在。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 94 | 模板选择错误 | 高 | FP16 分支错误使用 `GeluDAG<bfloat16_t>::OpDag`，应为 `GeluDAG<half>::OpDag` |
| 2 | 93-94 | 类型映射不一致 | 高 | dType 设为 TPL_FP16 但 DAG 模板使用 bfloat16_t，tiling key 与实际计算图不匹配 |
| 3 | 94 | 模板参数错误 | 高 | half 和 bfloat16_t 是不同数据类型，会导致计算图配置错误 |

> **核心问题总结**: 第 92-94 行的 FP16 分支存在一个关键 bug — 模板参数应为 `half`（float16_t）而非 `bfloat16_t`。这是一个复制粘贴错误（从第 96-97 行的 BF16 分支复制后未修改模板参数）。修复方法：将第 94 行改为 `baseTilingResult = elewiseBaseTiling.DoTiling<GeluDAG<half>::OpDag>(*tiling);`
