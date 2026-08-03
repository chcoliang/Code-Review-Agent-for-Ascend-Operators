# Ascend NPU 算子代码审查报告: aclnn_convolution.cpp

## Bug 列表

### Bug 1: ConvL0Warper 中逻辑运算符优先级错误导致条件永假

- **位置**: 第 153 行
- **类型**: 逻辑错误
- **严重程度**: 严重 (Critical)
- **描述**: 条件表达式 `opInfo.inputDtype == op::DataType::DT_FLOAT16 && opInfo.inputDtype == op::DataType::DT_BF16` 永远为 `false`，因为一个变量不可能同时等于两个不同的值。由于运算符优先级，`&&` 先于 `||` 求值，导致整个条件退化为仅检查 `DT_HIFLOAT8` 和 `DT_FLOAT8_E4M3FN`。`DT_FLOAT16` 和 `DT_BF16` 类型的输入将错误地走 `CONV_WITHFLAG_FUNCTION` 分支（带 `useHf32` 参数），而非预期的 `CONV_FUNCTION` 分支（无 `useHf32` 参数）。
- **触发条件**: 当 `inputDtype` 为 `DT_FLOAT16` 或 `DT_BF16` 时触发，会调用错误的函数指针类型进行 `reinterpret_cast`，可能导致参数错位、未定义行为甚至崩溃。
- **测试方案**: 使用 FP16 或 BF16 数据类型的 input/weight 调用卷积，在非 IsSupportND() 平台上验证是否正确调用不带 `useHf32` 的 L0 函数。

---

### Bug 2: All 函数递归调用了 Any 而非 All，语义错误

- **位置**: 第 266-273 行
- **类型**: 逻辑错误
- **严重程度**: 中等 (Medium)
- **描述**: `All` 函数的注释说明"参数需要满足所有参数列表判断"，但在递归时调用了 `Any(value, f, list...)` 而非 `All(value, f, list...)`。这意味着只有第一个元素需要满足条件，其余元素只需任一满足即可，违反了 `All` 的语义。
- **触发条件**: 当使用 `CHECK_PARAM_ALL_GTE`、`CHECK_PARAM_LT_ALL`、`CHECK_PARAM_GT_ALL` 宏且传入 3 个及以上比较参数时，第 2 个之后的参数只要有一个满足条件就会通过检查，无法正确拦截非法参数。
- **测试方案**: 构造一组参数如 `CHECK_PARAM_ALL_GTE(0, int64_t, 1, -1, 2)`，期望检查失败(因为 -1 < 0)，但由于 Any 语义实际会通过检查。

---

### Bug 3: CheckEmptyTensorTransposed 中逻辑条件永假

- **位置**: 第 1351 行
- **类型**: 逻辑错误
- **严重程度**: 中等 (Medium)
- **描述**: 条件 `if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))` 永远为 `false`。因为 `weightShape[i] < 0` 和 `weightShape[i] == 0` 不可能同时成立。正确的写法应为 `if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0))`。
- **触发条件**: 在转置卷积（transposed=true）且 SocVersion 为 ASCEND910_95 时，传入 weight 某维度为负值或为 0（但 output 对应维度非 0）时，本应报错但不会拦截，可能导致后续计算使用非法 shape。
- **测试方案**: 在 ASCEND910_95 平台使用 transposed=true，构造 weight shape 中某维度为 -1 的输入，验证是否能正确返回 ACLNN_ERR_PARAM_INVALID。

---

### Bug 4: REFLECTION_MODE 常量名与值语义不匹配

- **位置**: 第 67 行，使用于第 2311 行
- **类型**: 语义错误 / 功能错误
- **严重程度**: 中等 (Medium)
- **描述**: 常量命名为 `REFLECTION_MODE`（反射模式），但其值为 `"constant"`（常量填充模式）。该常量在 `CommonPreProcessC04` 的 `PadV3` 调用中使用（第 2311 行）。C04 分支的 padding 目的是将 channel 维度补零到 4，使用 `"constant"` 模式（零填充）实际上是正确的行为，但变量命名严重误导。如果开发者后续根据变量名修改值为 `"reflect"`，将导致 padding 行为错误。
- **触发条件**: 代码维护或重构时，开发者按照变量名理解其语义，将值改为 `"reflect"` 时会导致 C04 分支 weight padding 使用反射模式而非零填充，产生计算错误。
- **测试方案**: 检查 PadV3 调用时传入的 mode 参数，确认 C04 场景下 channel 维度补零是否使用正确的 padding 模式。

---

### Bug 5: ConvL0Warper 函数参数按值传递 map 导致性能问题

- **位置**: 第 130-131 行，以及第 192 行 `L0FuncWarperByOpType`
- **类型**: 性能缺陷
- **严重程度**: 低 (Low)
- **描述**: `ConvL0Warper` 和 `L0FuncWarperByOpType` 函数的第一个参数 `std::map<std::string, L0FUNCTION> l0Functions` 按值传递，每次调用都会拷贝整个 map。应改为 `const std::map<std::string, L0FUNCTION>& l0Functions`。
- **触发条件**: 每次卷积操作调用该函数时都会触发不必要的 map 拷贝，增加内存分配和拷贝开销。
- **测试方案**: 性能基准测试，对比修改前后的卷积 host 端耗时。

---

### Bug 6: isNotDMA 函数中 outputW 获取逻辑存在冗余且初始值可能错误

- **位置**: 第 2493-2496 行
- **类型**: 逻辑错误
- **严重程度**: 低 (Low)
- **描述**: `outputW` 初始赋值为 `output->GetViewShape().GetDim(2)`（即 H 维度），然后在 `outputSize == CONV_2D_DIM_SIZE` 时重新赋值为 `GetDim(3)`（W 维度）。但此函数仅在 `CanSwitchC04` 中被调用，而 `CanSwitchC04` 要求 input format 为 NCHW 且为 4 维，因此 output 也应为 4 维。如果 output 不是 4 维（例如某些异常路径），`outputW` 将使用 H 维度的值计算 `hoNum`，导致 L1 size 估算错误。
- **触发条件**: 在 output 维度不等于 4 的情况下调用 `isNotDMA` 函数（当前路径上不太可能触发，但代码防御性不足）。
- **测试方案**: 确认调用链中 output 维度一定为 4，或添加防御性检查。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第153行 | 逻辑错误 | 严重 | `&&` 应为 `\|\|`，导致 FP16/BF16 类型走错函数分支，可能 UB |
| 2 | 第266-273行 | 逻辑错误 | 中等 | `All` 递归调用 `Any`，破坏全部满足语义，参数校验不完整 |
| 3 | 第1351行 | 逻辑错误 | 中等 | `<0 && ==0` 永假，`&&` 应为 `\|\|`，无法检测非法 weight shape |
| 4 | 第67行 | 语义错误 | 中等 | `REFLECTION_MODE = "constant"` 名值矛盾，易误导维护者 |
| 5 | 第130、192行 | 性能缺陷 | 低 | map 按值传递导致不必要拷贝 |
| 6 | 第2493-2496行 | 逻辑错误 | 低 | outputW 初始值取错维度(H而非W)，对非4维场景无防御 |
