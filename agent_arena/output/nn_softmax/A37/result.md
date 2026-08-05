# SoftmaxV2 算子定义代码审查报告

## 审查文件
`softmax_v2_def.cpp`

---

### Bug 1: 输入dtype注册了不合理的 INT32 类型

- **位置**: 第23行，Input("x") 的 DataType 列表第2个元素 `ge::DT_INT32`
- **类型**: dtype注册错误
- **严重程度**: 高
- **描述**: Softmax 算子的数学定义为 `softmax(x_i) = exp(x_i) / sum(exp(x_j))`，涉及指数运算和浮点除法，输入必须为浮点类型。将 `ge::DT_INT32` 注册为合法输入类型在语义上不正确，且底层 kernel 实现通常不支持整型输入，会导致计算错误或运行时失败。正确应为 `ge::DT_FLOAT16`。
- **触发条件**: 用户传入 INT32 类型的 tensor 作为 SoftmaxV2 的输入时，框架不会拦截该非法输入，导致下发到 kernel 后行为未定义（计算结果错误或 AICore 异常）。
- **测试方案**: 构造 INT32 输入 tensor 调用 SoftmaxV2，观察是否报错或输出异常结果；对比将输入类型修正为 FLOAT16 后的正常输出。

---

### Bug 2: 输出dtype列表与输入dtype列表不匹配（位置对应关系错乱）

- **位置**: 第29行，Output("y") 的 DataType 列表 `{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}`
- **类型**: dtype注册错误
- **严重程度**: 高
- **描述**: Ascend 算子定义中，Input 和 Output 的 DataType 列表按位置一一对应，构成合法的 dtype 组合。当前注册的对应关系为：
  - 组合0: 输入 FLOAT → 输出 FLOAT (正确)
  - 组合1: 输入 INT32 → 输出 FLOAT16 (错误，输入类型本身就不合理)
  - 组合2: 输入 BF16 → 输出 BF16 (正确)
  - 组合3: 输入 FLOAT16 → 输出 FLOAT (错误)

  组合3中，输入为 FLOAT16 时输出注册为 FLOAT，这不是 softmax 的标准行为。标准 softmax 要求输入输出同 dtype（即 FLOAT16→FLOAT16）。虽然存在 `half_to_float` 属性，但该属性应在 kernel 内部通过逻辑分支处理，而非作为默认的 dtype 注册组合。正确的输出列表应为 `{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16}`。
- **触发条件**: 用户以 FLOAT16 输入调用 SoftmaxV2（half_to_float=false 时），期望输出 FLOAT16，但算子定义强制输出为 FLOAT，导致后续算子图中 dtype 推导不一致，引发图编译失败或精度问题。
- **测试方案**: 以 FLOAT16 输入调用 SoftmaxV2，检查输出 tensor 的 dtype 是否为 FLOAT16；验证在 half_to_float=false 条件下输出 dtype 是否与输入一致。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第23行 Input DataType | dtype注册错误 | 高 | 输入错误注册了 INT32 类型，softmax 不支持整型输入，应为 FLOAT16 |
| 2 | 第29行 Output DataType | dtype注册错误 | 高 | 输出dtype列表与输入不匹配：组合1(INT32→FLOAT16)和组合3(FLOAT16→FLOAT)对应关系错误，正确应为 FLOAT16→FLOAT16 |
