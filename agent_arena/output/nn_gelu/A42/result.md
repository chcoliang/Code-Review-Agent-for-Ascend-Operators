# Gelu 算子定义代码审查报告

**文件**: `gelu_def.cpp`

---

### Bug 1: 无效的 SoC 平台名称 "mc62cm12a"

- **位置**: 第 43 行 `this->AICore().AddConfig("mc62cm12a", aicoreConfig);`
- **类型**: 编译选项配置错误
- **严重程度**: 严重 (Critical)
- **描述**: `"mc62cm12a"` 不是任何有效的 Ascend NPU SoC 平台名称。Ascend 平台合法的 SoC 名称包括 `"ascend910b"`、`"ascend910_93"`、`"ascend310p"` 等。该字符串无法匹配任何已知硬件平台，会导致算子在目标平台上无法被调度执行，编译时可能无警告但运行时算子找不到对应的 AI Core 配置。
- **触发条件**: 在任何 Ascend 硬件平台上部署该算子时，`"mc62cm12a"` 对应的配置永远不会被匹配，若用户目标平台原本应由此行覆盖，则算子将无法运行。
- **测试方案**: 在目标 Ascend 平台上执行算子调用，检查是否能正确加载 AI Core 配置；使用 `atc` 工具编译模型时检查是否有平台不匹配的告警。

---

### Bug 2: 可疑的 SoC 平台名称 "ascend910_95"

- **位置**: 第 42 行 `this->AICore().AddConfig("ascend910_95", aicoreConfig);`
- **类型**: 编译选项配置错误
- **严重程度**: 高 (High)
- **描述**: `"ascend910_95"` 不是已知的标准 Ascend SoC 版本标识。已知的 Ascend 910 系列 SoC 名称包括 `"ascend910"`、`"ascend910b"`、`"ascend910_93"`（Atlas A2 训练系列）等。如果目标平台是 Atlas A2 训练卡，正确的名称应为 `"ascend910_93"`；如果是 Atlas 800I A2 推理卡，应为 `"ascend910b"`。使用错误的 SoC 名称会导致算子在目标硬件上无法被正确识别和调度。
- **触发条件**: 在 Ascend 910 系列硬件上部署算子时，平台名称不匹配导致算子编译或调度失败。
- **测试方案**: 确认目标部署硬件型号，使用对应的正确 SoC 名称重新编译；在目标硬件上执行 GELU 推理验证算子是否正常加载。

---

### Bug 3: opFile.value 指向错误的核函数文件名

- **位置**: 第 41 行 `.ExtendCfgInfo("opFile.value", "gelu_apt")`
- **类型**: 编译选项配置错误
- **严重程度**: 高 (High)
- **描述**: `opFile.value` 用于指定算子对应的核函数实现文件名。此处设置为 `"gelu_apt"`，但 GELU 算子的标准核函数文件名应为 `"gelu"`（与算子名对应）。如果实际的 kernel 实现文件名为 `gelu.cpp` 而非 `gelu_apt.cpp`，则框架在运行时无法找到对应的核函数实现，导致算子执行失败。
- **触发条件**: 运行时框架根据 `opFile.value` 查找核函数二进制时，因文件名不匹配而找不到实现，导致算子调用失败。
- **测试方案**: 检查实际的 kernel 实现文件名是否为 `gelu_apt.cpp`；若不是，将该值修正为实际文件名（通常为 `"gelu"`）；执行算子单元测试验证是否能正常调用核函数。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 43 行 | 编译选项配置 | 严重 | 无效 SoC 名称 `"mc62cm12a"`，非合法 Ascend 平台标识 |
| 2 | 第 42 行 | 编译选项配置 | 高 | 可疑 SoC 名称 `"ascend910_95"`，非标准平台标识 |
| 3 | 第 41 行 | 编译选项配置 | 高 | `opFile.value` 设为 `"gelu_apt"`，核函数文件名可能错误 |

---

## 修复建议

```cpp
// 第 41-43 行修正为:
aicoreConfig.DynamicCompileStaticFlag(false)
    .DynamicFormatFlag(false)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true)
    .NeedCheckSupportFlag(false)
    .PrecisionReduceFlag(true)
    .ExtendCfgInfo("opFile.value", "gelu");        // 修正: gelu_apt -> gelu
this->AICore().AddConfig("ascend910b", aicoreConfig);  // 修正: 使用正确的SoC名称
```

需根据实际目标硬件平台和核函数文件名进行确认后修正。
