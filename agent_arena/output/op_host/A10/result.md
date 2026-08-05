# Mul Tiling Arch35 代码审查报告

## Bug 列表

### Bug 1: PostTiling 对非 double 类型错误扣减 DCACHE_SIZE

- **位置**: 第 206 行 `PostTiling()` 函数
- **类型**: UB size 计算错误
- **严重程度**: 高
- **描述**: `PostTiling()` 无条件执行 `context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE))`，对所有数据类型都减去 32KB 的 DCACHE_SIZE。但在 `DoOpTiling()` 中，只有 `double` 类型路径（第 143 行）将 `DCACHE_SIZE` 作为 `extraSize` 传给 `DoTiling`，其余所有类型（float16、int32、int8 等）的 `extraSize` 默认为 0。这意味着非 double 类型的 tiling 计算基于完整 UB 大小，但实际可用内存被 PostTiling 减少了 32KB，导致 kernel 实际可用 UB 比 tiling 计算时假设的少 32KB，可能引发 UB 内存越界。
- **触发条件**: 使用任何非 double 数据类型（如 float16、int32、int8 等）执行 Mul 算子，且数据量足以填满 UB 时触发越界。
- **测试方案**: 使用 float16 类型构造大 tensor（使 tiling 分块接近 UB 容量上限），运行算子检查是否出现 UB 越界或计算结果异常。

---

### Bug 2: double 类型路径 DCACHE_SIZE 被双重扣减

- **位置**: 第 143 行 (`DoTiling` 调用传入 `extraSize=DCACHE_SIZE`) 与第 206 行 (`PostTiling` 再次减去 `DCACHE_SIZE`)
- **类型**: UB size 计算错误 / tiling 参数错误
- **严重程度**: 中
- **描述**: double 类型路径在 `DoTiling` 时传入 `extraSize = DCACHE_SIZE (32KB)`，broadcast tiling 框架会在计算分块时预留这 32KB。随后 `PostTiling` 又通过 `SetLocalMemorySize(ubSize_ - DCACHE_SIZE)` 再次减去 32KB。如果 `SetLocalMemorySize` 影响实际 kernel 可用内存，则 double 路径的有效 UB 被减少了 64KB（双重扣减），造成 UB 利用率下降、性能劣化。
- **触发条件**: 使用 double(DT_DOUBLE) 类型执行 Mul 算子。
- **测试方案**: 对比 double 类型 Mul 算子实际 UB 使用量与理论最大分块量，验证是否存在 32KB 的冗余保留。通过 profiling 观察内存利用率。

---

### Bug 3: Complex64 使用了错误的 OpDag 模板

- **位置**: 第 150-153 行
- **类型**: 算子逻辑错误 / tiling 参数错误
- **严重程度**: 高
- **描述**: `DT_COMPLEX64` (即 `std::complex<float>`，8 字节) 使用了 `MulOp<int64_t>::OpDag` 进行 tiling。虽然 complex64 和 int64_t 单元素大小相同（8 字节），但复数乘法语义为 `(a+bi)*(c+di) = (ac-bd)+(ad+bc)i`，需要额外的中间 buffer 和不同的计算 DAG。对比第 145-149 行 `DT_COMPLEX32` 使用了专门的 `MulComplex32Op<int32_t, int64_t>::OpDag`，`DT_COMPLEX64` 也应使用类似的 `MulComplex64Op` 专用模板。使用 `MulOp<int64_t>` 会导致：(1) tiling 计算未考虑复数乘法所需的额外临时 buffer；(2) kernel 按整数乘法执行，计算结果完全错误。
- **触发条件**: 使用 DT_COMPLEX64 类型输入执行 Mul 算子。
- **测试方案**: 构造 complex64 类型的 tensor 进行 Mul 运算，对比 CPU 参考结果，验证计算正确性。

---

### Bug 4: PostTiling 中 int64_t 到 uint32_t 截断及潜在下溢

- **位置**: 第 206 行
- **类型**: 数据类型安全 / 潜在下溢
- **严重程度**: 低
- **描述**: `ubSize_` 为 `int64_t` 类型，`DCACHE_SIZE` 为 `int64_t`。若 `ubSize_` 未正确初始化（例如 `GetPlatformInfo()` 异常路径未处理）或值小于 `DCACHE_SIZE`，则 `ubSize_ - DCACHE_SIZE` 产生负数，`static_cast<uint32_t>` 将其转换为极大的无符号数，导致 `SetLocalMemorySize` 设置异常值。
- **触发条件**: `GetPlatformInfo()` 获取 UB 大小失败或返回异常小值（< 32KB）时触发。
- **测试方案**: Mock `GetPlatformInfo` 返回小于 32KB 的值，验证 PostTiling 是否产生异常行为；或注入 platformInfo 为 null 且 compileInfo 也为 null 的场景。

---

### Bug 5: double 路径 extraBufferNum 为 0 与 extraSize 不匹配

- **位置**: 第 143 行
- **类型**: tiling 参数错误
- **严重程度**: 低
- **描述**: `DoTiling<MulDoubleOp<double>::OpDag>(context, tilingKey, DCACHE_SIZE, 0)` 传入 `extraSize=32KB` 但 `extraBufferNum=0`。在 BroadcastBaseTiling 框架中，`extraSize` 表示额外需要预留的内存大小，`extraBufferNum` 表示额外 buffer 数量。传入 extraSize 非零但 extraBufferNum 为 0 可能导致框架分配逻辑不一致（取决于框架实现），例如 extraSize 被忽略或计算错误。
- **触发条件**: 使用 double 类型执行 Mul 算子时，若框架依赖 extraBufferNum 来决定是否使用 extraSize。
- **测试方案**: 审查 BroadcastBaseTiling 源码确认 extraBufferNum=0 时 extraSize 的处理逻辑；使用 double 类型大 tensor 验证分块结果是否正确预留了 32KB。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | L206 | UB size 错误 | 高 | PostTiling 对所有类型无条件减 DCACHE_SIZE，非 double 路径 tiling 未预留该空间，导致潜在 UB 越界 |
| 2 | L143 + L206 | UB size 双重扣减 | 中 | double 路径 extraSize 和 PostTiling 各扣 32KB，实际多减 32KB，性能劣化 |
| 3 | L150-153 | 算子逻辑错误 | 高 | Complex64 错误使用 MulOp<int64_t> 而非专用复数乘法 OpDag，计算结果错误 |
| 4 | L206 | 数据安全 | 低 | int64_t 转 uint32_t 潜在下溢截断风险 |
| 5 | L143 | tiling 参数 | 低 | double 路径 extraSize 非零但 extraBufferNum=0，可能逻辑不一致 |
