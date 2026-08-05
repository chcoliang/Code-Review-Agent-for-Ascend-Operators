# Swish 算子定义代码审查报告

**文件**: `swish_def.cpp`  
**审查范围**: 算子定义、dtype注册、编译选项

---

### Bug 1: 输入数据类型注册错误 — DT_INT8 应为 DT_FLOAT16

**位置**: 第 24 行  
```cpp
.DataType({ ge::DT_INT8, ge::DT_BF16, ge::DT_FLOAT })
```

**类型**: dtype注册错误  
**严重程度**: 严重 (Critical)

**描述**:  
Input `x` 的第一个注册数据类型为 `ge::DT_INT8`（整型），而对应 Output `y` 的第一个数据类型为 `ge::DT_FLOAT16`。Swish 算子的数学定义为 `y = x * sigmoid(x)`，涉及 sigmoid 指数运算，INT8 整型数据无法直接进行此类浮点运算。从输入输出对应关系来看，第一组应为 `DT_FLOAT16 → DT_FLOAT16`，与第二组 `DT_BF16 → DT_BF16`、第三组 `DT_FLOAT → DT_FLOAT` 保持一致的同类型映射模式。

**触发条件**:  
当用户尝试以 INT8 类型的 tensor 调用 Swish 算子时，要么在图编译阶段因 kernel 实现不支持 INT8 而报错，要么在算子选择阶段匹配到此错误注册导致计算结果异常。

**测试方案**:  
1. 构造 INT8 类型输入 tensor，调用 Swish 算子，验证是否能正常编译和执行
2. 构造 FLOAT16 类型输入 tensor，验证算子是否无法匹配（因为 FLOAT16 未注册为合法输入类型）
3. 对比标准 Swish 算子规格，确认支持的数据类型列表

---

### Bug 2: 输入输出数据类型不对应

**位置**: 第 24 行（Input）与第 29 行（Output）  
```cpp
// Input:  DT_INT8,    DT_BF16, DT_FLOAT
// Output: DT_FLOAT16, DT_BF16, DT_FLOAT
```

**类型**: dtype注册错误  
**严重程度**: 严重 (Critical)

**描述**:  
Swish 是逐元素（element-wise）算子，其输出类型通常应与输入类型一致。第一组注册的输入类型为 `DT_INT8`、输出类型为 `DT_FLOAT16`，类型不匹配。正确的注册应为输入输出类型一一对应：`DT_FLOAT16→DT_FLOAT16, DT_BF16→DT_BF16, DT_FLOAT→DT_FLOAT`。

**触发条件**:  
框架在进行算子类型推导（InferDataType）时，输入 INT8 对应输出 FLOAT16 的映射会导致后续算子的输入类型与预期不符，可能引发整图类型推导失败或隐式类型转换开销。

**测试方案**:  
1. 在计算图中将 Swish 的输出连接到只接受与输入同类型的下游算子，验证类型推导是否报错
2. 单独验证第一组 dtype 配置下的精度和功能正确性

---

### Bug 3: 无效的 SoC 平台名称 "mc62cm12a"

**位置**: 第 42 行  
```cpp
this->AICore().AddConfig("mc62cm12a", aicoreConfig);
```

**类型**: 编译选项/平台配置错误  
**严重程度**: 严重 (Critical)

**描述**:  
`"mc62cm12a"` 不是有效的 Ascend SoC 平台名称。Ascend CANN 框架支持的合法平台名称格式如：`"ascend310"`, `"ascend310p"`, `"ascend910"`, `"ascend910b"` 等。此字符串无法匹配到任何实际硬件平台，会导致该配置完全无效，算子无法在目标平台上部署。

**触发条件**:  
在任何实际 Ascend 硬件上部署时，此配置行永远不会被匹配，相当于死代码。如果目标平台正是被此无效名称替代的平台，则算子在该平台上无法运行。

**测试方案**:  
1. 尝试在所有已知 Ascend 平台上编译部署该算子，检查 "mc62cm12a" 配置是否被加载
2. 检查 CANN 框架日志中是否有未识别平台名称的警告
3. 确认目标部署平台并使用正确的 SoC 名称替换

---

### Bug 4: 可疑的 SoC 平台名称 "ascend910_95"

**位置**: 第 41 行  
```cpp
this->AICore().AddConfig("ascend910_95", aicoreConfig);
```

**类型**: 编译选项/平台配置错误  
**严重程度**: 中等 (Medium)

**描述**:  
`"ascend910_95"` 不是常见的 Ascend SoC 标准命名。标准命名通常为 `"ascend910b"`（对应 Atlas 训练系列 910B）或 `"ascend910"` 等。如果目标平台为 Ascend 910B，正确的 SoC 名称应为 `"ascend910b"`。使用非标准名称可能导致平台匹配失败。

**触发条件**:  
在 Ascend 910B 硬件上部署算子时，如果框架无法识别 `"ascend910_95"` 则算子不会被正确调度。

**测试方案**:  
1. 查阅目标 CANN 版本支持的 SoC 名称列表，确认 "ascend910_95" 是否合法
2. 在目标平台上执行算子编译，验证配置是否生效
3. 如不合法，替换为正确的 SoC 名称（如 `"ascend910b"`）

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第24行 | dtype注册错误 | 严重 | 输入类型 DT_INT8 应为 DT_FLOAT16，INT8不支持Swish浮点运算 |
| 2 | 第24/29行 | dtype注册错误 | 严重 | 输入输出第一组类型不对应(INT8→FLOAT16)，应保持同类型映射 |
| 3 | 第42行 | 平台配置错误 | 严重 | "mc62cm12a" 非合法Ascend SoC平台名称，配置无效 |
| 4 | 第41行 | 平台配置错误 | 中等 | "ascend910_95" 非标准SoC命名，疑似应为 "ascend910b" |

---

## 修复建议

```cpp
// 修复后的关键代码段:
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT })  // DT_INT8 → DT_FLOAT16
    .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
    .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT })
    .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
    .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });

// 平台配置修复（需根据实际目标平台确认）:
this->AICore().AddConfig("ascend910b", aicoreConfig);  // 修正平台名称
```
