# Ascend NPU 算子代码审查报告

**文件**: `mul_tiling_arch35.cpp`
**审查范围**: Tiling参数、UB size相关Bug

---

### Bug 1: FP16分支使用了错误的OpDag模板

- **位置**: 第119-123行
- **类型**: Tiling参数错误（OpDag模板选择错误）
- **严重程度**: 高
- **描述**: `DT_FLOAT16` 输入输出组合使用了 `MulOp<float>::OpDag`，而非预期的 `MulXfp16Op<half>::OpDag`。对比第114-118行的 `DT_BF16` 分支正确使用了 `MulXfp16Op<bfloat16_t>::OpDag`。`MulOp<float>` 按照 float（4字节）计算element size进行tiling分块，但实际数据为 half（2字节），导致UB buffer利用率仅为50%，且tiling分块参数与实际kernel数据类型不匹配，可能导致计算结果错误或性能严重下降。
- **触发条件**: 输入输出均为 `DT_FLOAT16` 类型时触发。
- **测试方案**: 构造两个 float16 tensor 执行 Mul 算子，对比使用 `MulXfp16Op<half>::OpDag` 的正确结果；检查tiling参数中的blockSize是否按2字节对齐而非4字节。

---

### Bug 2: PostTiling对所有数据类型无条件减去DCACHE_SIZE，与非double类型的tiling计算不一致

- **位置**: 第210行
- **类型**: UB size计算不一致
- **严重程度**: 高
- **描述**: `PostTiling()` 中 `SetLocalMemorySize(ubSize_ - DCACHE_SIZE)` 对所有数据类型都减去了 32KB 的 DCACHE_SIZE。但在 `DoOpTiling()` 中，只有 `DT_DOUBLE` 分支（第147行）通过 `extraSize = DCACHE_SIZE` 参数告知tiling计算需要预留这32KB，其他所有数据类型的tiling计算（extraSize默认为0）假设可使用完整的UB空间。这导致非double类型的tiling分块是基于完整UB计算的，但kernel实际可用UB被PostTiling减少了32KB，可能导致UB buffer溢出（OOB写入）。
- **触发条件**: 任何非 `DT_DOUBLE` 数据类型，且数据量较大使得tiling接近UB容量上限时触发。
- **测试方案**: 使用float32类型、构造使tiling刚好铺满UB的大tensor，运行算子检查是否出现UB越界访问（可通过AscendCL的内存检测工具检测）；或对比tiling计算使用的UB size与PostTiling设置的localMemorySize是否一致。

---

### Bug 3: COMPLEX64使用MulOp<int64_t>而非复数乘法专用模板

- **位置**: 第154-157行
- **类型**: Tiling参数错误（OpDag模板选择错误）
- **严重程度**: 高
- **描述**: `DT_COMPLEX64`（由两个float32组成的复数，共8字节）使用了 `MulOp<int64_t>::OpDag`，将复数当作普通8字节整数进行逐元素乘法。正确的复数乘法需要 `(a+bi)*(c+di) = (ac-bd)+(ad+bc)i`，需要专用的复数乘法DAG。对比第149-153行的 `DT_COMPLEX32` 正确使用了 `MulComplex32Op` 专用模板。应使用类似 `MulComplex64Op` 的专用模板。
- **触发条件**: 输入输出均为 `DT_COMPLEX64` 类型时触发，计算结果完全错误。
- **测试方案**: 构造两个complex64 tensor（如 `(1+2i)*(3+4i)`），验证结果是否为 `(-5+10i)`；使用 `MulOp<int64_t>` 会得到错误的按位整数乘法结果。

---

### Bug 4: FP16与FLOAT32使用相同的OpDag模板导致tiling key冲突

- **位置**: 第119-127行
- **类型**: Tiling Key冲突
- **严重程度**: 中
- **描述**: `DT_FLOAT16`（第119-123行）和 `DT_FLOAT`（第124-128行）两种不同数据类型使用了完全相同的 `MulOp<float>::OpDag` 模板，生成的 `tilingKey` 将相同（因为 `GET_TPL_TILING_KEY` 基于 SchMode）。当系统缓存或复用tiling key时，可能导致fp16场景错误地使用float的kernel配置，或调度到错误的kernel实现。
- **触发条件**: 同一进程中先后执行fp16和float32的Mul算子，且框架基于tilingKey进行kernel调度。
- **测试方案**: 在同一session中交替执行fp16和float32的Mul算子，检查是否使用了正确的kernel函数和tiling参数。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 第119-123行 | Tiling参数错误 | 高 | FP16分支错误使用 `MulOp<float>::OpDag`，应使用 `MulXfp16Op<half>::OpDag` |
| 2 | 第210行 | UB size不一致 | 高 | PostTiling对所有类型减32KB，但仅double的tiling预留了该空间，其他类型存在UB越界风险 |
| 3 | 第154-157行 | Tiling参数错误 | 高 | COMPLEX64错误使用整数乘法模板，应使用复数乘法专用模板 |
| 4 | 第119-127行 | Tiling Key冲突 | 中 | FP16和FLOAT32使用相同OpDag导致tilingKey相同，可能引发调度错误 |
