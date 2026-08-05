# aclnn_matmul.cpp 代码审查报告

## Bug 列表

### Bug 1: `aclnnMatmulWeightNzGetWorkspaceSize` 中 Executor 资源泄漏

- **位置**: 第 929-935 行
- **类型**: 资源泄漏 (Resource Leak)
- **严重程度**: 高
- **描述**: 在 `aclnnMatmulWeightNzGetWorkspaceSize` 函数中，`CREATE_EXECUTOR()` 在第 930 行创建了 executor，但参数检查 `CheckWeightNzParam` 在第 934 行才执行。如果参数检查失败（返回非 SUCCESS），函数通过 `CHECK_RET` 直接返回错误码，此时已创建的 `uniqueExecutor` 未通过 `ReleaseTo(executor)` 释放给调用者，也没有其他释放路径（虽然智能指针会析构，但 `*executor` 未被赋值，调用者无法管理该资源）。对比 `aclnnMatmulGetWorkspaceSize`（第 887-888 行），该函数正确地在创建 executor 之前执行了参数检查。
- **触发条件**: 调用 `aclnnMatmulWeightNzGetWorkspaceSize` 时传入无效参数（如不支持的数据类型、不满足 NZ shape 约束等），使 `CheckWeightNzParam` 返回失败。
- **测试方案**: 传入 dtype 为 DT_FLOAT 的 mat2 tensor（NZ 格式不支持 float32），验证函数返回错误后 executor 指针仍为 nullptr 且无内存泄漏（通过内存检测工具如 ASan/valgrind 验证）。

---

### Bug 2: `const_cast` 修改 const 输入参数导致未定义行为和调用者状态被污染

- **位置**: 第 466 行, 第 794 行
- **类型**: 未定义行为 (Undefined Behavior) / 接口契约违反
- **严重程度**: 高
- **描述**: 在 `BuildMatMulWeightNzGraph`（第 466 行）和 `MatMulWeightNzGraph::Impl`（第 794 行）中，通过 `const_cast<aclTensor*>(mat2)->SetViewShape(...)` 修改了声明为 `const aclTensor*` 的输入参数的 ViewShape。这违反了 const 语义契约：(1) 如果调用者传入的对象确实是 const 的，修改它是 C++ 标准定义的未定义行为；(2) 即使对象本身非 const，调用者不会预期其输入被修改，后续使用 mat2 时 ViewShape 已被交换，导致逻辑错误。且修改后没有恢复原始 shape 的代码路径。
- **触发条件**: 当 mat2 为 NZ 格式且 stride 模式被判定为转置时（`GetTransposeAttrValue` 返回 true），调用 `aclnnMatmulWeightNzGetWorkspaceSize` 或新路径的 `MatMulWeightNzGraph::Impl`。
- **测试方案**: 创建一个转置的 NZ 格式 mat2，调用 matmul 后检查 mat2 的 ViewShape 是否被意外修改（与调用前比较）。多次连续调用同一 mat2 的 matmul，验证第二次调用是否因 shape 已被交换而产生错误结果。

---

### Bug 3: 常量命名误导 `NZ_K0_VALUE_32 = 8`

- **位置**: 第 53 行
- **类型**: 命名错误 (Naming Bug) / 可维护性缺陷
- **严重程度**: 低
- **描述**: 常量 `NZ_K0_VALUE_32` 的值为 8，而 `NZ_K0_VALUE_16` 的值为 16。命名模式暗示 "VALUE_32" 应该是值 32，但实际含义是 "32位数据类型(float32)对应的 K0 值为 8"。这种命名极易误导开发者，在后续维护中可能产生错误引用。若有人按命名直觉认为该值为 32 并在其他地方硬编码 32，将导致计算错误。
- **触发条件**: 后续开发者维护或扩展代码时，基于命名约定误用该常量。
- **测试方案**: 代码审查项；建议重命名为 `NZ_K0_VALUE_FOR_FP32 = 8` 或 `NZ_K0_VALUE_FLOAT32 = 8` 以消除歧义。

---

### Bug 4: `CheckShapeValid` 中条件分支注释与逻辑不匹配

- **位置**: 第 180 行
- **类型**: 逻辑缺陷 / 注释误导
- **严重程度**: 中
- **描述**: 第 180 行注释为 `"tensor1 dims number is 1 OR tensor2 dims number is 2"`，但实际条件是 `dimTensor2 == 1 || dimTensor2 == 2`（检查的是 tensor2 的维度为 1 或 2）。注释描述了完全不同的条件语义。虽然当前代码逻辑本身（对 dimTensor2==1 或 2 的情况取 mat2Shape.GetDim(0) 作为 K 轴）是正确的，但错误注释会误导后续维护者对逻辑进行错误修改。
- **触发条件**: 维护者根据注释理解逻辑并尝试"修正"代码时可能引入 bug。
- **测试方案**: 代码审查项；修正注释为 "tensor2 dims number is 1 OR 2"。

---

### Bug 5: `CreateMatmulGraphImpl` 第 845 行注释与条件完全颠倒

- **位置**: 第 845 行
- **类型**: 注释错误
- **严重程度**: 低
- **描述**: 注释写 `"dimTensor1 is 1 or 2 && dimTensor2 >= 3"`，但实际代码条件为 `dimTensor1 >= 3 && (dimTensor2 == 1 || dimTensor2 == 2)`，两者语义完全相反。这是 dimTensor1 和 dimTensor2 写反了。
- **触发条件**: 代码维护时根据注释理解分支逻辑。
- **测试方案**: 代码审查项；修正注释为 "dimTensor1 >= 3 && dimTensor2 is 1 or 2"。

---

### Bug 6: `extern "C"` 包裹含有 C++ 特性的命名空间和类定义

- **位置**: 第 43-45 行, 第 963-964 行
- **类型**: 语言规范违规 / 可移植性问题
- **严重程度**: 中
- **描述**: `extern "C"` 块（第 44 行开始）内部包含了匿名命名空间、类定义（`MatMulEmptyTensorGraph` 等）、`std::shared_ptr`、模板使用等纯 C++ 特性。虽然大多数编译器对此不会报错（`extern "C"` 主要影响链接符号的 name mangling），但标准上 `extern "C"` 内不应定义命名空间或类。更重要的是，导出的 C 接口函数（如 `aclnnMatmul`）确实需要 `extern "C"`，但内部实现不需要。正确做法是仅将对外 API 函数声明放在 `extern "C"` 块中。
- **触发条件**: 使用严格标准模式编译或移植到不同编译器时可能触发编译警告或错误。
- **测试方案**: 使用 `-pedantic` 编译选项检查是否有警告；重构为仅将 4 个对外 API 函数放在 `extern "C"` 块中。

---

### Bug 7: `MatMulDimNumMatBGe3Graph::PreProcess` 中 dimTensor1==2 时 unsqueeze 维度硬编码不合理

- **位置**: 第 734-738 行
- **类型**: 逻辑缺陷
- **严重程度**: 中
- **描述**: 当 `dimTensor1 == 2` 时，代码固定 unsqueeze 维度为 `{0, 1}`，将 2D tensor 变为 4D。但此分支处理的是 `dimTensor2 >= 3`（mat2 可能是 3D、4D、5D 或 6D）。当 mat2 为 3D 时，self 被扩展到 4D（比 mat2 多一个维度），batch 维度不匹配；当 mat2 为 5D 或 6D 时，self 仅被扩展到 4D，维度仍然不足。正确做法应根据 `dimTensor2` 动态确定需要 unsqueeze 的维度数量（即 `dimTensor2 - 2` 个前导维度）。
- **触发条件**: 当 self 为 2D (如 shape [n, m])，mat2 为 3D (如 shape [B, m, p]) 时，self 被 unsqueeze 为 [1, 1, n, m]（4D），与 3D 的 mat2 进行 BatchMatmul 时维度不匹配。或当 mat2 为 5D/6D 时维度不足。
- **测试方案**: 构造 self=[3,4] (2D), mat2=[2,4,5] (3D) 的输入，验证 matmul 是否能正确计算出 shape [2,3,5] 的结果。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第 929-935 行 | 资源泄漏 | 高 | Executor 在参数检查前创建，检查失败时泄漏 |
| 2 | 第 466, 794 行 | 未定义行为 | 高 | const_cast 修改 const 输入参数，污染调用者状态 |
| 3 | 第 53 行 | 命名错误 | 低 | NZ_K0_VALUE_32=8，命名与值严重不一致 |
| 4 | 第 180 行 | 注释错误 | 中 | 注释与实际条件语义不匹配 |
| 5 | 第 845 行 | 注释错误 | 低 | 注释中 dimTensor1/dimTensor2 完全写反 |
| 6 | 第 43-45 行 | 规范违规 | 中 | extern "C" 包裹 C++ 命名空间和类定义 |
| 7 | 第 734-738 行 | 逻辑缺陷 | 中 | unsqueeze 维度硬编码，未适配 mat2 的实际维度 |
