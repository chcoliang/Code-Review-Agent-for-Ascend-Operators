# Ascend NPU 算子代码审查报告 — swish_dag.h (A218)

## Bug 列表

### Bug 1: SwishNegOne DAG 内存层级配置错误

- **位置**: 第 153 行 `using MemCfg = MemOptCfg<MemLevel::LEVEL_1>;`
- **类型**: 内存层级 (Memory Level) 配置错误
- **严重程度**: 高 (High)
- **描述**: `SwishNegOne` DAG 使用了 `MemLevel::LEVEL_1`（L1 Buffer），而其内部的 `SwishNegOneDagCalc` 通过 `GetPhyAddr()` 获取 UB 地址并执行向量寄存器运算（Exp、Div 等）。向量计算单元只能访问 UB（Unified Buffer，对应 LEVEL_2）中的数据。对比 `SwishZero`（第 172 行）和 `SwishOther`（第 188 行）均正确使用了 `MemLevel::LEVEL_2`，此处应为笔误。使用 LEVEL_1 会导致 DAG 调度器将数据停留在 L1 层级，向量指令无法正确读写数据，产生运行时错误或计算结果异常。
- **触发条件**: 当 scale 参数为 -1 时进入 `SwishNegOne` 分支，任何输入 shape 均会触发。
- **修复建议**: 将第 153 行改为 `using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;`
- **测试方案**: 
  1. 构造 scale=-1 的用例，输入为随机 float/half tensor，验证输出是否等于 `x / (1 + exp(x))`。
  2. 对比修改前后在 Ascend 硬件或精确仿真器上的运行结果，修改前应报错或结果全零/异常。

---

### Bug 2: 循环内 Mask 计算未处理尾部迭代

- **位置**: 第 54、68、107、123 行 `mask = AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(count);`
- **类型**: DAG 逻辑 / 向量计算逻辑错误
- **严重程度**: 高 (High)
- **描述**: 在多次循环迭代中，`UpdateMask` 始终传入总元素数 `count`，而非当前迭代剩余待处理的元素数。当 `count > vl`（即 count > 64 for float）时，存在多次迭代。对于非尾部迭代，若 `UpdateMask` 对超过 vl 的值截断为全 1 mask 则行为正确；但对于**尾部迭代**（最后一轮），应只处理 `count - loopIdx * vlSize` 个元素。当前实现在尾部迭代仍使用 `count` 作为参数，如果 `UpdateMask` 实现为 `count % vl` 取余逻辑，则中间迭代的 mask 会错误地不满（只覆盖部分元素）；如果 `UpdateMask` 对 `count > vl` 返回全 1 mask，则尾部迭代会**越界读写**超出有效数据范围的 UB 内存。
- **触发条件**: `count` 不是 `vl`（64）的整数倍时，尾部迭代 mask 不正确。例如 count=100，最后一次迭代应只处理 36 个元素。
- **修复建议**: 将 mask 计算改为：
  ```cpp
  uint32_t remaining = (count > (loopIdx + 1) * vlSize) ? vlSize : (count - loopIdx * vlSize);
  mask = AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(remaining);
  ```
- **测试方案**:
  1. 构造 count 为非 64 对齐的输入（如 shape=[100]），检查输出 tensor 尾部元素是否正确。
  2. 检查是否有内存越界写入（在目标 buffer 后方填充 sentinel 值，验证是否被覆盖）。

---

### Bug 3: 非 float 类型时 vlSize 基于 float 计算导致地址偏移错误

- **位置**: 第 69、75、124、132 行 `src1Addr + loopIdx * vlSize` / `dstAddr + loopIdx * vlSize`
- **类型**: 内存层级 / 地址计算逻辑错误
- **严重程度**: 中 (Medium)
- **描述**: `vlSize` 按 `VECTOR_REG_WIDTH / sizeof(float) = 64` 计算，表示一个向量寄存器可容纳的 float 元素数。在非 float 分支（T=half/bfloat16）中，`src1Addr` 类型为 `T*`，指针算术 `src1Addr + loopIdx * vlSize` 每次偏移 64 个 T 元素（即 128 字节 for half）。然而使用 `DIST_UNPACK_B16` 加载时，从 UB 读取的是 half 数据并解包到 float 寄存器。一个 256-bit 向量寄存器解包 B16 数据时，实际从内存读取的是 64 个 half 元素（128 字节），偏移 64 个 T 元素是正确的。但写入时使用 `DIST_PACK_B32`，将 64 个 float 打包写回为 64 个 half 元素（128 字节）到 `dstAddr + loopIdx * vlSize`，这也是 64 个 T 偏移，逻辑正确。**经深入分析，此处地址偏移在语义上是正确的**（每次处理 64 个 T 元素），故降级为**潜在风险**而非确认 bug——但变量命名 `vlSize` 暗示 float 语义，在 T!=float 时容易引起误解，建议添加注释或使用更明确的命名。
- **严重程度修正**: 低 (Low) — 代码可工作但可维护性差
- **触发条件**: T 为 half/bfloat16 时。
- **测试方案**: 使用 half 类型输入验证多 loop 场景下各段数据计算结果是否正确。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 153 行 | 内存层级配置 | **高** | SwishNegOne 错误使用 LEVEL_1，应为 LEVEL_2 |
| 2 | 第 54/68/107/123 行 | DAG/向量逻辑 | **高** | UpdateMask 使用总 count 而非每次迭代剩余元素数，尾部迭代越界或 mask 错误 |
| 3 | 第 69/75/124/132 行 | 可维护性/命名 | **低** | vlSize 按 float 计算用于 T* 指针偏移，语义正确但命名易误导 |

## 总结

本文件存在 **2 个高严重度 bug**：
1. `SwishNegOne` 的内存层级错误（LEVEL_1 → LEVEL_2），会导致 scale=-1 场景下向量计算无法访问正确数据。
2. 循环内 mask 未按迭代更新剩余元素数，导致非对齐 count 场景下尾部数据计算错误或越界。

建议优先修复 Bug 1 和 Bug 2，均为功能性缺陷，会在特定输入条件下产生错误结果。
