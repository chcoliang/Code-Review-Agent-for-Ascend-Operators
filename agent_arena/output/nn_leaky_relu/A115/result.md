# LeakyRelu 算子定义代码审查报告

**文件**: `leaky_relu_def.cpp`

---

### Bug 1: 无效的 SoC 平台名称 "mc62cm12a"

- **位置**: 第 40 行 `this->AICore().AddConfig("mc62cm12a", aicoreConfig);`
- **类型**: 编译选项/平台配置错误
- **严重程度**: 严重 (Critical)
- **描述**: `"mc62cm12a"` 不是 Ascend NPU 的合法 SoC 平台标识符。合法的平台名称应为 `"ascend910b"`、`"ascend910_95"`、`"ascend310p"` 等规范命名。使用无效平台名将导致该配置无法被框架识别，算子在目标平台上无法部署或编译失败。
- **触发条件**: 当用户尝试在该无效平台名对应的真实硬件上加载或编译此算子时，框架将找不到匹配的 AICore 配置，导致算子注册失败或运行时报错。
- **测试方案**: 在各目标 Ascend 平台上执行算子编译和注册，验证 `AddConfig` 中的平台名是否能被框架正确识别；检查编译日志中是否出现 "unsupported soc" 类警告或错误。

---

### Bug 2: negative_slope 属性默认值错误 (0.0 应为 0.01)

- **位置**: 第 30 行 `this->Attr("negative_slope").AttrType(OPTIONAL).Float(0.0);`
- **类型**: 算子定义/语义错误
- **严重程度**: 严重 (Critical)
- **描述**: LeakyReLU 的标准定义中，`negative_slope` 的默认值应为 `0.01`（参考 PyTorch `torch.nn.LeakyReLU` 和 ONNX LeakyRelu 算子规范）。当前默认值设为 `0.0`，会导致算子在用户未显式指定该属性时退化为普通 ReLU（负区间输出恒为 0），与 LeakyReLU 的语义不一致，产生静默的计算错误。
- **触发条件**: 用户调用 LeakyRelu 算子时不传入 `negative_slope` 参数（依赖默认值），输出结果将与标准框架（PyTorch/ONNX）的 LeakyReLU 结果不一致。
- **测试方案**: 构造含负值的输入张量，不指定 `negative_slope`，对比本算子输出与 PyTorch `F.leaky_relu(x)` 的输出，验证负区间是否保留 0.01 倍的斜率而非归零。

---

### Bug 3: PrecisionReduceFlag 设置为 true 可能导致精度问题

- **位置**: 第 37 行 `.PrecisionReduceFlag(true)`
- **类型**: 编译选项/精度配置
- **严重程度**: 中等 (Medium)
- **描述**: `PrecisionReduceFlag(true)` 允许框架在编译时降低计算精度以换取性能。对于 LeakyReLU 算子，当 `negative_slope` 值较小（如 0.01）时，精度降低可能导致负区间的微小斜率被截断或丢失，从而在 BF16/FP16 低精度场景下输出与预期偏差较大。LeakyReLU 属于逐元素简单运算，不存在累积误差，不应启用精度降低。
- **触发条件**: 使用 BF16 或 FP16 数据类型，且 `negative_slope` 值较小时，精度降低可能导致负区间输出为 0 或误差超出容忍范围。
- **测试方案**: 在 BF16/FP16 模式下，使用较小 negative_slope（如 0.001），输入含大量负值的张量，对比开启/关闭 PrecisionReduceFlag 时的输出精度差异，验证最大绝对误差是否在可接受范围内。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第 40 行 | 编译选项/平台配置 | 严重 | 无效 SoC 名称 "mc62cm12a"，非合法 Ascend 平台标识 |
| 2 | 第 30 行 | 算子定义/语义错误 | 严重 | negative_slope 默认值为 0.0，应为 0.01，导致退化为 ReLU |
| 3 | 第 37 行 | 编译选项/精度配置 | 中等 | PrecisionReduceFlag(true) 对简单逐元素算子不应启用，可能损失小斜率精度 |
