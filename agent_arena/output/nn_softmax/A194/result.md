# SoftmaxV2 Base Tiling 代码审查报告

## Bug 列表

### Bug 1: UB Size 计算在两条路径中不一致

- **位置**: `GetPlatformInfo()` 函数，第 245 行 vs 第 252 行
- **类型**: UB size 计算错误
- **严重程度**: 严重 (Critical)
- **描述**: 当 `platformInfoPtr == nullptr` 时（第245行），`aicoreParams_.ubSize` 直接使用 `compileInfo->ubSize`，而 `compileInfo->ubSize` 在 `TilingPrepareForSoftmaxV2AscendC`（第281行）中被赋值为完整的 UB 大小（`ubSizeTemp`，未做除法）。但当 `platformInfoPtr != nullptr` 时（第252行），`aicoreParams_.ubSize` 被赋值为 `ubSizeTemp / 4`（仅为 UB 总量的 1/4）。同一硬件上两条执行路径计算出的可用 UB 大小不同，导致 tiling 参数不一致。
  - 路径1（无 platformInfo）: `ubSize = 完整UB`
  - 路径2（有 platformInfo）: `ubSize = 完整UB / 4`
- **触发条件**: 当运行时 `context_->GetPlatformInfo()` 返回非空指针时（正常执行流），UB 可用空间被计算为实际的 1/4；若返回空指针（fallback 路径），则使用完整 UB 大小。两条路径的 tiling 结果截然不同，可能导致在 fallback 路径中因超出实际可用 UB 而产生内存越界。
- **测试方案**: 
  1. 构造单元测试分别模拟 platformInfoPtr 为 null 和非 null 两种场景，对比 `aicoreParams_.ubSize` 的值
  2. 在 fallback 路径下运行大 shape softmax 算子，检查是否触发 UB 溢出
  3. 对比两条路径下 tiling 的切分结果是否一致

---

### Bug 2: OP_LOGD 第一个参数类型错误

- **位置**: `TilingPrepareForSoftmaxV2()` 函数，第 316 行
- **类型**: 接口调用参数错误
- **严重程度**: 中等 (Medium)
- **描述**: `OP_LOGD(context, "TilingPrepareForSoftmaxV2AscendC enter");` 第一个参数传入了 `context`（`gert::TilingParseContext*` 类型），但根据本文件中其他所有 `OP_LOGD` 的用法（如第260、243、247行），第一个参数应为 `context->GetNodeName()` 返回的节点名字符串。传入错误类型可能导致编译警告、日志输出乱码，或在某些宏实现下产生未定义行为。
- **触发条件**: 每次调用 `TilingPrepareForSoftmaxV2` 时都会触发
- **测试方案**:
  1. 编译时开启 -Wall -Werror 检查是否有类型不匹配告警
  2. 运行后检查日志输出是否正常打印节点名称

---

### Bug 3: 循环变量类型与比较对象不匹配（潜在截断）

- **位置**: `GetDimsAndCheckShapeValid()` 第 154 行，`GetShapeAttrsInfo()` 第 214 行
- **类型**: 类型安全问题
- **严重程度**: 低 (Low)
- **描述**: 循环变量 `i` 声明为 `int`（32位），但 `xShapeSize_` 为 `int64_t`（64位）。当 `xShapeSize_` 超过 `INT_MAX` 时（虽然实际维度不会这么大），`i < xShapeSize_` 的比较会产生隐式符号转换。尽管维度已限制为 MAX_DIMS（8），但代码风格上不规范，且如果将来 MAX_DIMS 约束被移除可能暴露问题。
- **触发条件**: 当前不会触发实际问题（因 MAX_DIMS=8），但属于编码规范缺陷
- **测试方案**:
  1. 编译时开启 `-Wsign-compare` 检查告警
  2. 静态分析工具扫描

---

### Bug 4: reduceAxes_ 默认值设置时机依赖调用顺序

- **位置**: `GetAndCheckAxes()` 第 178 行
- **类型**: 隐式依赖/健壮性
- **严重程度**: 低 (Low)
- **描述**: 当 `axisListPtr` 为空时，`reduceAxes_ = xShapeSize_ - 1`。该赋值依赖 `xShapeSize_` 已在 `GetDimsAndCheckShapeValid()` 中正确初始化。当前 `GetShapeAttrsInfo()` 中调用顺序正确，但如果未来代码重构调换了函数调用顺序，将导致使用未初始化的 `xShapeSize_`，产生未定义行为。缺乏防御性检查。
- **触发条件**: 当前不会触发（调用顺序正确），但重构时有风险
- **测试方案**:
  1. 在 `GetAndCheckAxes` 入口处添加 `xShapeSize_ > 0` 的断言
  2. 单独调用 `GetAndCheckAxes()` 验证是否会使用垃圾值

---

## 汇总表

| 编号 | 位置 (行号) | Bug 类型 | 严重程度 | 简要描述 |
|------|-------------|----------|----------|----------|
| 1 | 245 / 252 / 281 | UB size 计算错误 | 严重 | 两条路径计算 ubSize 不一致：fallback用完整UB，正常路径用1/4 UB，可能导致内存越界 |
| 2 | 316 | 接口参数类型错误 | 中等 | OP_LOGD 传入 context 指针而非 GetNodeName()，日志异常或UB |
| 3 | 154 / 214 | 类型不匹配 | 低 | int vs int64_t 循环变量，符号比较隐患 |
| 4 | 178 | 隐式依赖 | 低 | reduceAxes_ 默认值依赖 xShapeSize_ 已初始化，缺防御检查 |

## 关键修复建议

**Bug 1 修复** (最高优先级):
```cpp
// GetPlatformInfo() 第252行，应与 TilingPrepare 保持一致
// 方案A：如果 /4 是正确的预留策略，则 TilingPrepare 中也应存储 ubSize/4
compileInfoPtr->ubSize = static_cast<int64_t>(ubSizeTemp / 4);  // line 281

// 方案B：如果应使用完整UB，则 GetPlatformInfo 不应除4
aicoreParams_.ubSize = static_cast<int64_t>(ubSizeTemp);  // line 252
```

**Bug 2 修复**:
```cpp
// 第316行
OP_LOGD(context->GetNodeName(), "TilingPrepareForSoftmaxV2AscendC enter");
```
