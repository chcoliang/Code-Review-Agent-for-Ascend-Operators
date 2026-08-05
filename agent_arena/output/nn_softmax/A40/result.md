# Softmax V2 Base Tiling 代码审查报告

**文件**: `softmax_v2_base_tiling.cpp`
**审查范围**: Tiling参数计算、UB size计算相关bug

---

### Bug 1: UB Size 错误地乘以2

- **位置**: 第281行
  ```cpp
  compileInfoPtr->ubSize = static_cast<int64_t>(ubSizeTemp) * 2;
  ```
- **类型**: UB size计算错误
- **严重程度**: 严重 (Critical)
- **描述**: `GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp)` 已经返回了单个核的完整UB物理大小。将其乘以2会导致tiling阶段认为可用UB空间是实际的两倍，后续kernel在UB上分配buffer时会超出物理UB边界，造成内存越界写入和数据损坏。
- **触发条件**: 任何调用 `TilingPrepareForSoftmaxV2AscendC` 的场景均会触发。当kernel基于该膨胀的ubSize计算tiling参数（如每次处理的数据量），分配的buffer总量超过实际UB容量时，运行时会出现UB越界访问。
- **测试方案**: 
  1. 构造一个shape较大的softmax用例（如shape=[64, 50000]，axis=-1），使得tiling计算需要接近占满UB的buffer分配。
  2. 开启AscendCL的内存检测工具（如`ASCEND_GLOBAL_LOG_LEVEL=1`配合msprof内存检查），观察是否报UB越界错误。
  3. 对比修复前后（去掉`* 2`）的tiling结果和kernel运行精度。

---

### Bug 2: blockDim类型转换不一致

- **位置**: 第244行 vs 第249行
  ```cpp
  // 第244行 (从compileInfo路径):
  aicoreParams_.blockDim = static_cast<int32_t>(compileInfo->coreNum);
  // 第249行 (从platform路径):
  aicoreParams_.blockDim = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
  ```
- **类型**: Tiling参数类型不一致
- **严重程度**: 中等 (Medium)
- **描述**: 同一个成员 `aicoreParams_.blockDim` 在两个分支中被分别用 `int32_t` 和 `int64_t` 进行强制转换赋值。如果 `blockDim` 的声明类型为 `int32_t`，则第249行的 `static_cast<int64_t>` 转换没有意义（隐式截断回int32_t）；如果声明类型为 `int64_t`，则第244行会产生窄化截断。这种不一致表明至少有一个分支的类型转换存在错误，可能导致核数设置异常。
- **触发条件**: 当 `platformInfoPtr != nullptr`（走第249行路径）且核数值超过 `INT32_MAX` 时（理论场景）会出现截断；或当 `blockDim` 为 `int32_t` 类型时，两条路径的语义虽然运行正确但代码意图不清晰，增加维护风险。
- **测试方案**:
  1. 检查 `aicoreParams_.blockDim` 的声明类型，确认应统一使用 `int32_t` 或 `int64_t`。
  2. 分别走两条路径（有/无platformInfo），对比最终 blockDim 值是否一致。

---

### Bug 3: OP_LOGD 第一参数类型错误

- **位置**: 第316行
  ```cpp
  OP_LOGD(context, "TilingPrepareForSoftmaxV2AscendC enter");
  ```
- **类型**: 日志接口调用错误
- **严重程度**: 中等 (Medium)
- **描述**: `OP_LOGD` 宏的第一个参数应传入节点名称字符串（`const char*` 类型），此处直接传入了 `context` 指针（`gert::TilingParseContext*` 类型）。对比同文件其他调用（第260、297、300行）均使用 `context->GetNodeName()`。传入错误类型可能导致编译告警、日志输出乱码，或在某些宏实现中引发未定义行为。
- **触发条件**: 每次执行 `TilingPrepareForSoftmaxV2` 函数时触发。在开启DEBUG日志级别时会输出错误内容。
- **测试方案**:
  1. 开启DEBUG级别日志，执行softmax算子，检查该行日志是否输出乱码或崩溃。
  2. 使用 `-Wall -Werror` 编译选项检查是否产生类型不匹配的编译错误/告警。
  3. 修复为 `OP_LOGD(context->GetNodeName(), "TilingPrepareForSoftmaxV2AscendC enter");`。

---

## 汇总表

| 编号 | 位置 | Bug类型 | 严重程度 | 简要描述 |
|------|------|---------|----------|----------|
| 1 | 第281行 | UB size计算错误 | 严重 | UB大小被错误地乘以2，导致kernel可能UB越界 |
| 2 | 第244/249行 | Tiling参数类型不一致 | 中等 | blockDim赋值在两个分支使用不同的类型转换 |
| 3 | 第316行 | 日志接口调用错误 | 中等 | OP_LOGD第一参数应为GetNodeName()而非context指针 |
