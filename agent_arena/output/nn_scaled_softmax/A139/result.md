# Code Review: aclnn_scaled_masked_softmax.cpp (A139)

## Bug 列表

### Bug 1: DT_INT32 不应出现在 Softmax 输入支持的数据类型列表中

- **位置**: 第 35 行
- **类型**: 逻辑错误 / 数据类型校验缺陷
- **严重程度**: 高
- **描述**: `SOFTMAX_X_DTYPE_SUPPORT_LIST` 包含 `op::DataType::DT_INT32`。Softmax 运算涉及指数运算和除法，对整型数据执行没有数学意义，且底层 kernel 通常不支持整型输入。这会导致计算结果完全错误或 kernel 执行失败。
- **触发条件**: 用户传入 dtype 为 INT32 的 tensor 作为 x，校验通过后进入 kernel 层执行。
- **测试方案**: 构造一个 INT32 类型的 4D tensor 作为 x 输入，验证是否产生计算错误或异常。

---

### Bug 2: 错误日志中 DIM_3 范围描述硬编码为 4096，未反映实际动态限制

- **位置**: 第 106 行
- **类型**: 日志/信息不一致
- **严重程度**: 低
- **描述**: 当平台为 ASCEND910_95 时，`dDimLimit` 实际为 8192，但错误日志始终输出 `"Expected x and mask dim4 in range of (0, 4096]"`，对用户产生误导。应使用 `dDimLimit` 变量动态拼接日志信息。
- **触发条件**: 在 ASCEND910_95 平台上，传入 DIM_3 = 5000 的 tensor（合法范围内但超出 4096），日志仍提示上限为 4096。
- **测试方案**: 在 ASCEND910_95 平台上传入 DIM_3 > 8192 的 tensor，检查错误日志中的范围提示是否为 8192。

---

### Bug 3: 未校验输出 tensor y 与输入 x 的 shape 一致性

- **位置**: 第 75-111 行 (`CheckShape` 函数)
- **类型**: 校验遗漏
- **严重程度**: 高
- **描述**: `CheckShape` 只校验了 x 和 mask 之间的 shape 关系，但未验证输出 tensor y 的 shape 是否与 x 一致。如果 y 的 shape 与 x 不匹配，kernel 写入时可能发生越界访问或数据截断。
- **触发条件**: 用户传入 shape 不匹配的 y tensor（如 y 的某个维度小于 x），进入 kernel 后发生内存越界写入。
- **测试方案**: 构造 x shape=[2,4,8,16] 而 y shape=[1,4,8,16] 的 tensor 对，验证是否能检测到 shape 不匹配。

---

### Bug 4: namespace 匿名空间中使用 `extern` 声明外部函数导致链接问题

- **位置**: 第 39-44 行
- **类型**: 编码规范/潜在链接错误
- **严重程度**: 中
- **描述**: `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize` 和 `aclnnInnerScaledMaskedSoftmaxV2` 的 `extern` 声明位于匿名 namespace 内。匿名 namespace 赋予其内容内部链接属性（internal linkage），而 `extern` 期望外部链接（external linkage），两者语义冲突。虽然多数编译器在 `extern "C"` 块内会优先使用外部链接而绕过此问题，但这属于未定义行为的边缘地带，在某些编译器/标准版本下可能产生链接失败。
- **触发条件**: 使用严格遵循标准的编译器编译时，可能报错或产生链接问题。
- **测试方案**: 使用 `-pedantic` 编译选项编译，检查是否有警告或错误。

---

### Bug 5: `fixedTriuMask` 参数仅支持 false 但接口仍暴露该参数，且传入 true 时仅记日志未给出完整信息

- **位置**: 第 132-135 行
- **类型**: 接口设计/校验不完整
- **严重程度**: 低
- **描述**: 当 `fixedTriuMask = true` 时，仅打印错误日志并返回错误码，但日志拼写有误（"suppport" 多了一个 p），且未向用户说明后续支持计划或替代方案。这是一个拼写错误，可能影响日志检索。
- **触发条件**: 用户传入 `fixedTriuMask = true`。
- **测试方案**: 传入 `fixedTriuMask = true`，检查返回值和日志内容拼写。

---

## 汇总表

| # | 位置 | 类型 | 严重程度 | 简述 |
|---|------|------|----------|------|
| 1 | L35 | 逻辑错误 | 高 | Softmax 支持类型列表错误包含 DT_INT32 |
| 2 | L106 | 日志不一致 | 低 | 错误提示硬编码 4096，未适配 8192 场景 |
| 3 | L75-111 | 校验遗漏 | 高 | 未校验输出 y 与输入 x 的 shape 一致性 |
| 4 | L39-44 | 链接语义冲突 | 中 | 匿名 namespace 内 extern 声明语义矛盾 |
| 5 | L133 | 拼写错误 | 低 | "suppport" 拼写错误，影响日志检索 |
