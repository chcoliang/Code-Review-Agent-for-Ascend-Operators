# Swish Tiling Arch35 代码审查报告

## Bug 列表

### Bug 1: FP16 路径下 NEG_ONE 分支模板类型错误

- **位置**: 第 132 行
- **类型**: 模板类型映射错误
- **严重程度**: 严重 (Critical)
- **描述**: 当 `outputDtype == ge::DT_FLOAT16` 且 `attrWork == TPL_SCALE_NEG_ONE` 时，代码错误地使用了 `SwishDag::SwishNegOne<bfloat16_t>::OpDag` 模板实例化，而正确的类型应该是 `SwishDag::SwishNegOne<half>::OpDag`。对比同一 `DT_FLOAT16` 分支内的 ZERO 和 OTHER 路径（第 134、136 行）均正确使用了 `half` 类型，以及 `DT_BF16` 分支（第 139-144 行）使用 `bfloat16_t` 的模式，可以确认第 132 行是将 `half` 误写为 `bfloat16_t`。
- **触发条件**: 输入数据类型为 float16，且 scale 属性值为 -1.0 时触发。会导致 kernel 使用错误的数据类型进行计算，可能产生精度错误或内存访问越界。
- **修复建议**: 将第 132 行改为:
  ```cpp
  baseTilingResult = elewiseBaseTiling.DoTiling32B<SwishDag::SwishNegOne<half>::OpDag>();
  ```
- **测试方案**: 构造 float16 输入 tensor，设置 scale=-1.0，执行 Swish 算子，对比 CPU 参考实现结果验证精度。

---

### Bug 2: schMode 变量未在本文件中定义或初始化

- **位置**: 第 163 行
- **类型**: Tiling 参数错误（潜在未初始化变量）
- **严重程度**: 中等 (Medium)
- **描述**: `GET_TPL_TILING_KEY(schMode, attrWork)` 中使用了 `schMode` 变量，但在本文件中该变量既未声明也未赋值。如果它是 `SwishTiling` 类的成员变量且未在构造函数或 `RunTiling` 流程中显式初始化，则其值为不确定值（未初始化内存），会导致生成错误的 tilingKey，进而导致 kernel 选择错误或运行时失败。
- **触发条件**: 当 `SwishTiling` 类的 `schMode` 成员未被正确初始化时，所有调用路径都会受影响。
- **修复建议**: 确认 `schMode` 在头文件 `swish_tiling_arch35.h` 中的声明及初始化逻辑；如缺失则应在 `RunTiling` 中根据平台/调度策略进行正确赋值。
- **测试方案**: 在不同硬件配置和调度模式下运行 Swish 算子，检查 tilingKey 是否与预期匹配；使用内存检测工具（如 ASan）检测未初始化变量读取。

---

### Bug 3: 常量 ZERO 使用 double 类型字面量

- **位置**: 第 37 行
- **类型**: 类型不一致（轻微）
- **严重程度**: 低 (Low)
- **描述**: `ZERO` 定义为 `0.0`（double 类型），而 `NEG_ONE` 定义为 `-1.0f`（float 类型）。在第 106 行 `scale == ZERO` 比较时，float 变量 `scale` 会隐式提升为 double 进行比较。虽然对于零值比较不会产生精度问题，但类型不一致违反编码规范，且若将来修改为其他非精确表示的浮点数可能引入隐患。
- **触发条件**: 编译期类型提升，运行时对零值比较无实际影响，但存在代码规范问题。
- **修复建议**: 将第 37 行改为:
  ```cpp
  static constexpr float ZERO = 0.0f;
  ```
- **测试方案**: 编译时开启 `-Wdouble-promotion` 警告，验证修复后无警告。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 132 行 | 模板类型映射错误 | 严重 | FP16+NEG_ONE 分支错误使用 bfloat16_t 而非 half |
| 2 | 第 163 行 | Tiling 参数错误 | 中等 | schMode 变量来源不明，可能未初始化 |
| 3 | 第 37 行 | 类型不一致 | 低 | ZERO 常量为 double 类型，应为 float |
