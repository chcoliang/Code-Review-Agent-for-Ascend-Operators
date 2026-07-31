# A194 代码审查报告 - softmax_v2_base_tiling.cpp

## Bug 1: UB Size 在不同路径下计算不一致

- **位置**: 第 252 行 vs 第 245-246 行
- **类型**: Tiling 参数计算错误
- **严重程度**: 高
- **描述**: 当 `platformInfoPtr` 不为空时（第 252 行），`aicoreParams_.ubSize` 被设置为 `ubSizeTemp / 4`；但当 `platformInfoPtr` 为空时（第 245-246 行），`aicoreParams_.ubSize` 直接使用 `compileInfo->ubSize`，而 `compileInfo->ubSize` 在 `TilingPrepareForSoftmaxV2AscendC`（第 281 行）中被赋值为完整的 `ubSizeTemp`（未除以 4）。两条路径对 UB size 的处理不一致，通过 compile info 路径得到的 UB size 是 platform 路径的 4 倍。
- **触发条件**: 当 `platformInfoPtr` 为 nullptr 时走 compile info 路径，此时 ubSize 为实际值未除以 4，可能导致 tiling 计算分配的 UB 空间超出实际可用空间。
- **测试方案**: 
  1. 构造 platformInfoPtr 为 nullptr 的场景运行 tiling 计算
  2. 对比两条路径产生的 ubSize 值
  3. 使用大 shape 触发 UB 分配接近上限的情况，验证是否发生越界

## Bug 2: OP_LOGD 第一个参数类型错误

- **位置**: 第 316 行
- **类型**: 接口调用参数错误
- **严重程度**: 低
- **描述**: `OP_LOGD(context, "TilingPrepareForSoftmaxV2AscendC enter")` 中第一个参数应为 node name 字符串，但传入了 `context` 指针（`gert::TilingParseContext*` 类型）。同文件其他位置（如第 260、297、310 行）均使用 `context->GetNodeName()` 作为第一个参数。
- **触发条件**: 每次执行 `TilingPrepareForSoftmaxV2` 时都会触发，可能导致日志输出乱码或编译警告。
- **测试方案**: 
  1. 编译时检查是否有类型不匹配警告
  2. 运行时检查该行日志输出是否为乱码

---

## 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 252 vs 245 | Tiling参数计算 | 高 | UB Size 两条路径不一致，compile info 路径缺少除以4 |
| 2 | 316 | 接口调用错误 | 低 | OP_LOGD 第一个参数应为 GetNodeName() 而非 context 指针 |
