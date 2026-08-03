# Ascend NPU 算子代码审查报告

**文件**: `aclnn_convolution.cpp`  
**审查日期**: 2026-08-03

---

### Bug 1: `All()` 模板函数逻辑错误 — 递归调用 `Any()` 而非 `All()`

- **位置**: 第 265-273 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: `All()` 函数本应验证所有参数都满足条件，但在第一个比较通过后，递归调用了 `Any()` 而非 `All()`。这导致只要第一个参数满足条件且剩余参数中任意一个满足条件即返回 true，而非全部满足。
  ```cpp
  // 第 269 行: 应为 return All(value, f, list...);
  if (result) {
      return Any(value, f, list...);  // BUG
  }
  ```
- **触发条件**: 当使用 `CHECK_PARAM_ALL_GTE`、`CHECK_PARAM_LT_ALL`、`CHECK_PARAM_GT_ALL` 宏校验 3 个及以上参数时，若第一个参数合法但后续某些参数非法，校验会被错误跳过。
- **测试方案**: 构造含 3+ 个空间维度值的 tensor，使第一个维度合法而后续维度非法（如负数），验证校验是否正确拒绝。

---

### Bug 2: `CheckEmptyTensorTransposed` 中条件表达式永假

- **位置**: 第 1351 行
- **类型**: 逻辑错误
- **严重程度**: 高
- **描述**: 条件 `if (weightShape[i] < 0 && (weightShape[i] == 0 && outputShape[i] != 0))` 恒为 false，因为 `weightShape[i] < 0` 和 `weightShape[i] == 0` 不可能同时成立。正确逻辑应为 `||`（或运算）。
  ```cpp
  // 应改为:
  if (weightShape[i] < 0 || (weightShape[i] == 0 && outputShape[i] != 0))
  ```
- **触发条件**: 在 ASCEND910_95 平台上，transposed 模式下 weight 的非首维为负数或为 0（但 output 对应维度非 0）时，校验无法正确拦截非法输入。
- **测试方案**: 在 910_95 平台执行 transposed conv，weight shape 中包含负维度（如 [64, -1, 3, 3]），验证是否返回 ACLNN_ERR_PARAM_INVALID。

---

### Bug 3: `CommonPreProcess` 中 input 未执行 Contiguous 操作

- **位置**: 第 2217-2219 行
- **类型**: 数据处理缺陷
- **严重程度**: 高
- **描述**: 当 `contiguous=true` 时，weight 和 bias 均调用 `l0op::Contiguous()` 转为连续存储，但 input 仅做了 `contiguousInput = input` 的赋值，未调用 Contiguous。对比同文件 `CommonPreProcessC04`（第 2270 行）可确认遗漏。
  ```cpp
  if (contiguous) {
      contiguousInput = input;  // BUG: 缺少 l0op::Contiguous(input, executor)
      CHECK_RET(contiguousInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  ```
- **触发条件**: 当输入 tensor 的存储为非连续（如转置后的 tensor、stride 不规则的 view）时，后续 L0 算子接收到非连续数据可能导致计算错误或崩溃。
- **测试方案**: 使用 `torch.as_strided` 创建非连续 input tensor，执行 Conv2d，对比连续输入的结果。

---

### Bug 4: `REFLECTION_MODE` 常量命名与值不一致

- **位置**: 第 67 行
- **类型**: 语义错误
- **严重程度**: 中
- **描述**: 常量命名为 `REFLECTION_MODE`（反射模式），但实际值为 `"constant"`（常量填充模式）。该常量在第 2311 行 `PadV3` 调用中使用。如果上下文确实需要 constant padding 则功能正确但命名误导；如果需要 reflection padding 则为功能性 bug。
  ```cpp
  static const std::string REFLECTION_MODE = "constant";  // 名称暗示 reflection 但值为 constant
  ```
- **触发条件**: C04 分支中 weight padding 时使用此模式，若预期 reflection padding 行为则输出结果错误。
- **测试方案**: 对 Cin<4 的小通道输入执行 C04 分支卷积，检查 weight padding 补 0 行为是否符合预期。

---

### Bug 5: `ConvL0Warper` 和 `L0FuncWarperByOpType` 按值传递 map

- **位置**: 第 130 行、第 191 行
- **类型**: 性能缺陷
- **严重程度**: 中
- **描述**: 两个函数的 `std::map<std::string, L0FUNCTION> l0Functions` 参数按值传递，每次调用都会完整复制整个 map。应使用 `const std::map<std::string, L0FUNCTION>&` 引用传递。
  ```cpp
  // 第 130 行: 应改为 const std::map<std::string, L0FUNCTION>& l0Functions
  static const aclTensor* ConvL0Warper(
      std::map<std::string, L0FUNCTION> l0Functions, ...)
  ```
- **触发条件**: 每次卷积算子执行时均触发，造成不必要的内存分配和复制开销。
- **测试方案**: 性能测试对比修复前后的 GetWorkspaceSize 耗时。

---

### Bug 6: `Conv3dTo2dImpl` 中 `l0Functions` 成员变量遮蔽基类

- **位置**: 第 3692 行
- **类型**: 代码缺陷 / 潜在逻辑错误
- **严重程度**: 低
- **描述**: `Conv3dTo2dImpl` 类重新声明了 `std::map<std::string, L0FUNCTION> l0Functions;` 成员，遮蔽了基类 `ConvolutionImpl` 中的同名成员（第 3236 行）。虽然当前实现因 PreProcess/Impl 均在派生类中执行而功能上无误，但基类成员永远为空，若后续维护中基类方法引用 l0Functions 将导致找不到注册的函数。
- **触发条件**: 当前不直接触发功能问题，但代码维护时易引入 bug。
- **测试方案**: 在 310P 平台上执行 D=1 的 Conv3d（触发 Conv3dTo2d 路径），验证是否正常运行。

---

## 汇总表

| # | 位置 | 类型 | 严重程度 | 简要描述 |
|---|------|------|----------|----------|
| 1 | L265-273 | 逻辑错误 | 高 | `All()` 递归误调 `Any()`，参数校验被削弱 |
| 2 | L1351 | 逻辑错误 | 高 | `< 0 && == 0` 永假条件，transposed weight 校验失效 |
| 3 | L2217-2219 | 数据处理缺陷 | 高 | input 未做 Contiguous，非连续 tensor 可能计算错误 |
| 4 | L67 | 语义错误 | 中 | REFLECTION_MODE 命名与 "constant" 值矛盾 |
| 5 | L130, L191 | 性能缺陷 | 中 | map 按值传递导致不必要的深拷贝 |
| 6 | L3692 | 代码缺陷 | 低 | 派生类 l0Functions 遮蔽基类同名成员 |
