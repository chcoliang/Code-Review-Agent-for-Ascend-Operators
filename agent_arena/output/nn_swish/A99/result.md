# Swish Tiling Arch35 代码审查报告

## Bug 列表

### Bug 1: schMode 成员变量未初始化即使用

- **位置**: 第 163 行 `const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, attrWork);`
- **类型**: Tiling 参数错误
- **严重程度**: 高
- **描述**: `schMode` 在 `RunTiling()` 函数中被用于生成 tilingKey，但在当前文件中未见任何初始化赋值。若头文件 `swish_tiling_arch35.h` 中未提供默认初始化，该成员变量将包含未定义值，导致生成错误的 tilingKey，进而选择错误的 kernel 执行路径。
- **触发条件**: 任何调用 `RunTiling()` 的场景，若类构造函数或头文件未对 `schMode` 进行初始化。
- **测试方案**: 构造 SwishTiling 对象后直接调用 RunTiling()，检查生成的 tilingKey 是否在合法范围内；使用 valgrind/sanitizer 检测未初始化内存读取。

---

### Bug 2: WORKSPACE_SIZE 常量定义但未使用，实际 Workspace 大小注释与值不一致

- **位置**: 第 34-35 行
- **类型**: Tiling 参数 / UB Size 相关
- **严重程度**: 中
- **描述**: `WORKSPACE_SIZE = 32` 被定义但从未使用。实际使用的 `ASCEND_WORKSPACE = 16777216 * 2`（即 32MB），注释写为 `// 16 * 1024 * 1024` 仅解释了乘法因子之一，容易误导维护者认为 workspace 为 16MB。此外，32MB 的 workspace 对于 Ascend 310/910 的 GM workspace 来说可能过大或不必要，需确认硬件规格。
- **触发条件**: 维护者依据 WORKSPACE_SIZE=32 或注释判断实际 workspace 大小时产生误解；在 workspace 受限的硬件平台上运行时可能申请失败。
- **测试方案**: 在不同 Ascend 硬件平台上运行算子，检查 workspace 申请是否成功；验证 WORKSPACE_SIZE 是否为残留代码。

---

### Bug 3: ZERO 常量类型为 double，与 float 变量比较存在隐式类型提升

- **位置**: 第 37 行 `static constexpr float ZERO = 0.0;` 及第 106 行 `scale == ZERO`
- **类型**: 类型映射错误
- **严重程度**: 低
- **描述**: `ZERO` 赋值为 `0.0`（double 字面量），虽然变量声明为 `float` 所以实际存储为 float，但与第 36 行 `NEG_ONE = -1.0f`（显式 float 字面量）风格不一致。更关键的是，若后续有人将 ZERO 改为 `constexpr double`，则 `scale == ZERO` 会将 float 提升为 double 进行比较，可能因精度问题导致比较失败，使得 scale=0 的路径走入 `TPL_SCALE_OTHER` 分支。
- **触发条件**: 代码重构时若修改 ZERO 的类型声明；当前代码因 float 类型声明实际无功能影响。
- **测试方案**: 确认 `scale=0.0` 时是否正确进入 `TPL_SCALE_ZERO` 分支；静态分析工具检查隐式类型转换警告。

---

### Bug 4: SetScalar 在 DoTiling 之后调用，可能导致 Tiling 数据序列化顺序错误

- **位置**: 第 160 行 `elewiseBaseTiling.SetScalar<float>(attrScale);`
- **类型**: Tiling 参数错误
- **严重程度**: 高
- **描述**: `SetScalar<float>(attrScale)` 在 `DoTiling32B()` 之后调用。在 ElewiseBaseTiling 框架中，DoTiling 通常负责计算并序列化 tiling 数据到 buffer。如果 SetScalar 需要在 DoTiling 之前设置以便纳入 tiling 计算（如影响 UB buffer 分配中需要为 scalar 预留空间），则此处调用顺序错误，会导致 kernel 端读取到错误的 scalar 值或 tiling 参数。
- **触发条件**: 当 scale != 1.0 且走入 `TPL_SCALE_OTHER` 分支时，kernel 侧需要使用 scalar 值进行计算，若 tiling data 中 scalar 未正确写入则计算结果错误。
- **测试方案**: 使用 scale=2.0 的 Swish 算子，对比 CPU 结果与 NPU 结果，验证 scalar 值是否被正确传递到 kernel；dump tiling data buffer 检查 scalar 字段位置和值。

---

### Bug 5: 未使用的常量定义造成代码冗余

- **位置**: 第 29-33 行 `OP_KEY_INVALID`, `OP_KEY_1`, `OP_KEY_2`, `OP_KEY_3`, `INDEX_0`
- **类型**: 代码质量
- **严重程度**: 低
- **描述**: 五个常量 (`OP_KEY_INVALID`, `OP_KEY_1`, `OP_KEY_2`, `OP_KEY_3`, `INDEX_0`) 定义后在整个文件中从未被引用。这些残留代码增加维护负担，且可能暗示 tilingKey 的生成逻辑曾经使用固定 key 映射，后来改为 `GET_TPL_TILING_KEY` 宏但未清理旧代码。
- **触发条件**: 不会直接导致运行时错误，但在代码审计和重构时造成困惑。
- **测试方案**: 编译时开启 `-Wunused-variable` 确认警告；删除后验证功能不受影响。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | L163 | Tiling参数 | 高 | schMode 未初始化，tilingKey 可能为随机值 |
| 2 | L34-35 | UB Size/Workspace | 中 | WORKSPACE_SIZE 未使用，ASCEND_WORKSPACE 注释误导，32MB 可能过大 |
| 3 | L37 | 类型映射 | 低 | ZERO 使用 double 字面量赋值，风格不一致有隐患 |
| 4 | L160 | Tiling参数 | 高 | SetScalar 在 DoTiling 后调用，scalar 可能未正确序列化 |
| 5 | L29-33 | 代码质量 | 低 | 5个常量定义未使用，疑似残留代码 |
