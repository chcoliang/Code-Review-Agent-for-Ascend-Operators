# aclnn_leaky_relu.cpp 代码审查报告

## Bug 列表

### Bug 1: 输出张量 `out` 未进行空指针检查

- **位置**: 第 37-42 行，`CheckNotNull` 函数
- **类型**: 空指针解引用
- **严重程度**: 高
- **描述**: `CheckNotNull` 函数接收 `out` 参数但使用 `(void)out;` 显式跳过了空指针检查。后续代码在第 63 行 `CheckShape` 中直接访问 `out`（`OP_CHECK_SHAPE_NOT_EQUAL(out, self, ...)`），以及第 108 行访问 `out->GetDataType()`、第 112 行 `ViewCopy(castOut, out, ...)` 都会导致空指针解引用崩溃。
- **触发条件**: 调用 `aclnnLeakyReluGetWorkspaceSize` 时传入 `out = nullptr`。
- **测试方案**: 构造测试用例，传入有效的 `self` 和 `negativeSlope`，但 `out` 设为 `nullptr`，验证是否返回 `ACLNN_ERR_PARAM_NULLPTR` 而非崩溃。

### Bug 2: `negativeSlope->ToFloat()` 对 DT_DOUBLE 类型输入造成精度丢失

- **位置**: 第 104 行
- **类型**: 精度丢失 / 数值错误
- **严重程度**: 中
- **描述**: `negativeSlope` 标量通过 `ToFloat()` 转为单精度浮点数传给 LeakyRelu kernel。当输入张量数据类型为 `DT_DOUBLE` 时，negativeSlope 的高精度值被截断为 float32，导致计算结果精度下降。支持列表中明确包含 `DT_DOUBLE`，因此该路径会被触发。
- **触发条件**: 输入 `self` 的数据类型为 `DT_DOUBLE`，且 `negativeSlope` 的值需要超过 float32 精度才能准确表示（例如极小值 `1e-30` 或有效位数超过7位的值）。
- **测试方案**: 构造 DT_DOUBLE 类型输入，negativeSlope 设为需要双精度表示的值（如 `0.123456789012345`），对比输出与参考实现的精度差异。

### Bug 3: 输出张量 `out` 的数据类型未做合法性校验

- **位置**: 第 54-59 行，`CheckDtypeValid` 函数；第 67-78 行，`CheckParams` 函数
- **类型**: 校验缺失
- **严重程度**: 中
- **描述**: `CheckDtypeValid` 仅校验了输入 `self` 的数据类型是否在支持列表内，但未校验输出 `out` 的数据类型。如果 `out` 的数据类型为不支持的类型（如 INT32），第 108 行的 `Cast` 操作可能失败或产生未定义行为。
- **触发条件**: 调用时传入 `out` 的 dtype 为 `DT_INT32`、`DT_INT8` 等非浮点类型。
- **测试方案**: 构造输入为 DT_FLOAT 的 `self`，输出 `out` 设为 DT_INT32 类型，验证是否正确报错。

### Bug 4: Inplace 操作与 Cast 逻辑冲突

- **位置**: 第 121-124 行，`aclnnInplaceLeakyReluGetWorkspaceSize` 函数；第 108 行 Cast 操作
- **类型**: 逻辑错误
- **严重程度**: 低
- **描述**: Inplace 版本将 `selfRef` 同时作为输入和输出传入。在执行流程中，输入先被 Contiguous 化生成新 tensor，再经 LeakyRelu 计算，然后做 Cast（self 的类型 cast 到 self 的类型，即无意义的 Cast），最后 ViewCopy 回 `selfRef`。虽然功能上不会出错，但多了一次不必要的 Cast 调用，浪费计算资源。如果 LeakyRelu kernel 内部改变了输出类型（例如 float16 输入提升为 float32 计算），则 Cast 回 float16 可能造成精度下降，而 inplace 语义要求保持原始精度。
- **触发条件**: 使用 inplace 版本且输入为 DT_FLOAT16/DT_BF16（kernel 可能内部提升精度）。
- **测试方案**: 对 DT_FLOAT16 输入执行 inplace LeakyRelu，对比非 inplace 版本（输出设为 DT_FLOAT）的结果精度。

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简述 |
|------|------|----------|----------|------|
| 1 | 第 40 行 (`CheckNotNull`) | 空指针解引用 | 高 | `out` 参数未做空指针检查，后续解引用导致崩溃 |
| 2 | 第 104 行 (`ToFloat()`) | 精度丢失 | 中 | DT_DOUBLE 输入时 negativeSlope 被截断为 float32 |
| 3 | 第 54-59 行 (`CheckDtypeValid`) | 校验缺失 | 中 | 未校验输出 `out` 的数据类型合法性 |
| 4 | 第 121-124 行 (Inplace + Cast) | 逻辑冗余/精度风险 | 低 | Inplace 场景下存在无意义 Cast，低精度类型有精度下降风险 |
