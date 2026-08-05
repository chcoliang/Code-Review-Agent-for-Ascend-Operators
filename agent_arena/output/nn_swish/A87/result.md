# Ascend NPU 算子代码审查报告 - aclnn_swish.cpp (A87)

## Bug 列表

### Bug 1: executor 传参类型错误

- **位置**: 第 108 行
- **类型**: 接口调用错误 / 类型不匹配
- **严重程度**: 高 (High)
- **描述**: `ReshapeSelfValueGetActivation(self, dimSize, selfContiguous, uniqueExecutor)` 中最后一个参数传入了智能指针对象 `uniqueExecutor` 本身，而非原始指针 `uniqueExecutor.get()`。对比同文件中其他所有 l0op 调用（第103、115、120、123行）均使用 `uniqueExecutor.get()` 获取原始指针。传入智能指针对象会导致编译错误或隐式类型转换产生未定义行为。
- **触发条件**: 所有非空张量输入且维度 > 0 的正常调用路径均会触发。
- **测试方案**: 使用任意合法输入调用 `aclnnSwishGetWorkspaceSize`，验证是否能正确编译和执行；若编译通过（隐式转换），检查 executor 内部状态是否正确记录了 reshape 操作。

---

### Bug 2: reshapeSelf 缺少空指针检查

- **位置**: 第 108 行（使用在第 115 行）
- **类型**: 缺少错误处理 / 空指针解引用风险
- **严重程度**: 高 (High)
- **描述**: `ReshapeSelfValueGetActivation` 的返回值 `reshapeSelf` 没有进行空指针检查，直接传入第 115 行的 `l0op::Swish(reshapeSelf, scale, ...)`。若内部 reshape 失败返回 nullptr，将导致空指针解引用或后续算子行为异常。对比第 104 行对 `selfContiguous` 和第 116 行对 `swishOut` 都有 `CHECK_RET(... != nullptr, ...)` 检查。
- **触发条件**: 当输入张量维度超过 `MAX_SUPPORT_DIMS_NUMS`，内部 reshape 计算失败时触发（如内存不足等异常场景）。
- **测试方案**: 构造超高维张量（维度数 > MAX_SUPPORT_DIMS_NUMS）输入，在内存受限条件下执行，检查是否能安全返回错误码而非崩溃。

---

### Bug 3: reshape 输入使用原始 tensor 而非 contiguous tensor

- **位置**: 第 108 行
- **类型**: 逻辑错误
- **严重程度**: 中 (Medium)
- **描述**: `ReshapeSelfValueGetActivation` 的第一个参数传入了 `self`（原始非连续张量），而非 `selfContiguous`（第103行已做连续化处理的张量）。从函数语义看，该函数应基于连续化后的张量进行 reshape 判断和操作。传入原始 `self` 可能导致基于非连续内存布局做出错误的 reshape 决策，或者在后续计算中使用了未连续化的 tensor 元信息。
- **触发条件**: 当输入 tensor 为非连续存储（如经过 slice、transpose 等操作后的 view tensor）且维度超限需要 reshape 时触发。
- **测试方案**: 构造非连续张量（如 `tensor.transpose(0,1)` 后不调用 contiguous），传入 `aclnnSwish`，对比输出与 PyTorch `F.silu()` 参考结果的正确性。

---

### Bug 4: shapeOriDetial 取自 selfContiguous 而非原始 self 的 shape

- **位置**: 第 107 行
- **类型**: 逻辑错误
- **严重程度**: 低 (Low)
- **描述**: `GetTensorShapeActivation(selfContiguous, ...)` 从 contiguous 后的张量获取原始shape信息用于后续恢复维度（第120行）。通常 contiguous 不会改变 shape，但在某些带 format 转换的场景下（如 FRACTAL_NZ 等私有格式），contiguous 可能改变逻辑视图的 shape 表示。应使用 `self` 的原始 ViewShape 来记录需要恢复的目标形状。
- **触发条件**: 输入 tensor 为私有格式（如 FRACTAL_NZ/NC1HWC0）且维度超过 MAX_SUPPORT_DIMS_NUMS 时，恢复 shape 可能与预期不符。
- **测试方案**: 以私有格式的超高维张量为输入，验证输出 tensor 的 shape 是否与预期 `out` 的 shape 一致。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第108行 | 接口调用/类型错误 | 高 | `uniqueExecutor` 应为 `uniqueExecutor.get()`，传参类型不匹配 |
| 2 | 第108/115行 | 空指针风险 | 高 | `reshapeSelf` 返回值未做 nullptr 检查即使用 |
| 3 | 第108行 | 逻辑错误 | 中 | reshape 函数第一参数应传 `selfContiguous` 而非原始 `self` |
| 4 | 第107行 | 逻辑错误 | 低 | 恢复shape应基于原始 `self` 而非 `selfContiguous` |
