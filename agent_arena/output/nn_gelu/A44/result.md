# Gelu Tiling Arch35 代码审查报告

## Bug 列表

### Bug 1: UB Size 计算错误 - 错误地将 UB 大小翻倍

- **位置**: 第 138 行 `compileInfoPtr->ubSize = compileInfoPtr->ubSize * 2;`
- **类型**: UB size 计算错误
- **严重程度**: 致命 (Critical)
- **描述**: `GetCoreMemSize(platform_ascendc::CoreMemType::UB, ...)` 已经返回了单核实际可用的 UB 物理内存大小。将其乘以 2 会导致 tiling 计算时认为可用 UB 空间是实际的两倍，从而在 `DoTiling` 阶段分配超出物理 UB 容量的 buffer，运行时产生 UB 内存越界访问，导致数据损坏或硬件异常。
- **触发条件**: 任何输入 tensor 大小足以使单次处理数据量超过实际 UB 容量的一半时触发。即当实际需要的 buffer 大于真实 UB size 的 50% 时，翻倍后的错误 tiling 不会再切分，直接越界。
- **测试方案**: 构造一个较大的输入 tensor（例如元素数 > UB_SIZE / sizeof(dtype) / 2），对比正确 tiling（不乘2）和当前 tiling 的 blockDim 及每核处理量；在硬件上运行观察是否产生 `AIC_ERROR` 或输出数据错误。

---

### Bug 2: BF16 类型映射在 Arch35 平台上可能不受支持

- **位置**: 第 33 行（`CalcInputDtype` 允许 `ge::DT_BF16`）及第 96-97 行（`GeluDAG<bfloat16_t>` 实例化）
- **类型**: 类型映射 / 平台兼容性错误
- **严重程度**: 高 (High)
- **描述**: Arch35 对应的硬件平台（如 Ascend310P）的向量计算单元不支持原生 BF16 计算。代码中 `CalcInputDtype` 允许 `DT_BF16` 通过校验，并在 `RunTiling` 中实例化 `GeluDAG<bfloat16_t>` 进行 tiling 计算。当该算子部署到 Arch35 硬件时，kernel 中的 BF16 向量指令将无法执行，产生非法指令异常。
- **触发条件**: 用户在 Arch35 平台上以 BF16 数据类型调用 Gelu 算子时触发。
- **测试方案**: 在 Arch35 环境中构造 BF16 输入 tensor 调用 Gelu 算子，验证是否在编译期或运行期报错；对比 Arch35 硬件 spec 确认 BF16 指令集支持情况。

---

### Bug 3: Tiling Key 第一个参数与 DAG 实际配置可能不匹配

- **位置**: 第 112 行 `const uint64_t tilingKey = GET_TPL_TILING_KEY(1, dType);`
- **类型**: Tiling 参数错误
- **严重程度**: 中 (Medium)
- **描述**: `GET_TPL_TILING_KEY` 的第一个参数通常表示模板 pipeline 编号或算子变体编号，需要与 kernel 侧注册的 tiling key 严格一致。此处硬编码为 `1`，如果 `gelu_dag.h` 中 DAG 定义使用了不同的 pipeline 编号（如 0 或根据 dtype 变化），则 host 侧生成的 tilingKey 与 device 侧注册的 key 不匹配，导致 kernel 启动失败（找不到对应的 tiling 分支）。
- **触发条件**: 当 kernel 侧注册的 tiling key 第一个参数不是 1 时，所有调用均会失败。
- **测试方案**: 检查 `gelu_dag.h` 中 `OpDag` 的 pipeline 编号定义；运行算子观察是否返回 "tiling key not found" 类错误。

---

### Bug 4: 成员变量 `dType` 未显式初始化

- **位置**: 第 93/96/99 行赋值，第 112 行使用
- **类型**: 代码健壮性 / 潜在未定义行为
- **严重程度**: 低 (Low)
- **描述**: 类成员变量 `dType` 在构造函数中未显式初始化。虽然当前逻辑保证在到达第 112 行时 `dType` 已被赋值（else 分支会提前 return），但如果后续维护者修改了分支逻辑（如去掉 else return），`dType` 可能以未初始化状态被使用，产生未定义行为。
- **触发条件**: 当前代码路径不会触发，但代码修改后可能暴露。
- **测试方案**: 静态分析工具检查；在构造函数中将 `dType` 初始化为无效值，添加使用前断言。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 138 行 | UB size 计算错误 | 致命 | UB 大小错误乘以 2，导致 buffer 越界 |
| 2 | 第 33, 96-97 行 | 类型映射错误 | 高 | Arch35 不支持 BF16 但代码未拦截 |
| 3 | 第 112 行 | Tiling 参数错误 | 中 | tilingKey 硬编码参数可能与 kernel 不匹配 |
| 4 | 第 93/96/99 行 | 代码健壮性 | 低 | 成员变量 dType 未显式初始化 |
