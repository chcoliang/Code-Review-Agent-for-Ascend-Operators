# Swish DAG 代码审查报告

## Bug 列表

### Bug 1: SwishNegOne DAG CopyOut 类型不匹配

- **位置**: 第 150 行
  ```cpp
  using OpCopyOut = Bind<Vec::CopyOut<float>, Placeholder::Out0<float>, OpResult>;
  ```
- **类型**: 类型错误 (Type Mismatch)
- **严重程度**: 严重 (Critical)
- **描述**: `SwishNegOne` DAG 中 `OpCopyOut` 使用了 `Vec::CopyOut<float>` 和 `Placeholder::Out0<float>`，硬编码为 `float` 类型。但 `SwishNegOneDagCalc<T>` 继承自 `Vec::ElemwiseBinaryOP<T, T, float>`，其输出类型为 `T`。当 `T` 为 `half` 或 `bfloat16` 时，DAG 的 CopyOut 期望 `float` 类型数据，但实际计算节点输出为 `T` 类型（在非 float 路径中，第 74-75 行已将结果 Cast 回 `T` 并以 `T` 类型存储），导致类型不匹配。对比 `SwishOther`（第 185 行）正确使用了 `Vec::CopyOut<T>` 和 `Placeholder::Out0<T>`。
- **触发条件**: 当模板参数 `T` 为 `half` 或 `bfloat16` 等非 float 类型，且 `scale = -1` 时触发。
- **测试方案**: 使用 `half` 类型输入，`scale=-1` 调用 Swish 算子，对比 CPU 参考结果，验证输出数据类型和数值正确性。

---

### Bug 2: 循环中 Mask 未按剩余元素数更新

- **位置**: 第 54、68、107、123 行
  ```cpp
  mask = AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(count);
  ```
- **类型**: DAG 逻辑错误 (Logic Error)
- **严重程度**: 严重 (Critical)
- **描述**: 在多次循环迭代中，`UpdateMask` 始终传入总元素数 `count`，而非当前迭代应处理的剩余元素数。当 `count` 不是 `vl`（向量长度 64）的整数倍时，最后一次迭代的 mask 应仅覆盖 `count - loopIdx * vlSize` 个剩余元素。但由于 `count` 未更新，可能导致：(1) 如果 `UpdateMask(count)` 对 `count >= vl` 时生成全 mask，则最后一轮会越界读写超出有效数据范围的内存；(2) 如果 `UpdateMask` 取模运算 `count % vl`，则除最后一轮外其他迭代的 mask 不正确。正确写法应为：
  ```cpp
  uint32_t remaining = count - loopIdx * vlSize;
  mask = AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(remaining);
  ```
- **触发条件**: 当输入元素总数 `count` 不是 64 的整数倍且 `count > 64` 时触发。
- **测试方案**: 使用 `count = 65, 100, 127` 等非 64 对齐的元素个数进行测试，检查最后若干元素的输出是否正确，以及是否有内存越界（可用 sanitizer 检测）。

---

### Bug 3: 非 float 路径地址偏移与数据类型不一致（潜在风险）

- **位置**: 第 69、75、124、132 行
  ```cpp
  (__ubuf__ T*)(src1Addr + loopIdx * vlSize)
  ```
- **类型**: 精度/逻辑风险 (Potential Logic Error)
- **严重程度**: 中等 (Medium)
- **描述**: `vlSize` 基于 `sizeof(float)=4` 计算得到 64，表示每次处理 64 个 float 宽度的数据。对于非 float 类型 `T`（如 `half`），指针算术 `src1Addr + loopIdx * 64` 以 `T` 为单位偏移 64 个元素。由于使用 `DIST_UNPACK_B16` 加载，每次确实从内存中读取 64 个 `half` 元素并解包为 64 个 float 进行运算，逻辑上自洽。但此处 `vlSize` 的语义与 `vl`（基于 float 宽度）绑定，而指针类型为 `T*`，当 `T` 的大小与假设不符时（如未来支持 int8 等类型），偏移将出错。建议显式使用基于 `T` 元素个数的变量名或计算方式以提高可维护性。
- **触发条件**: 当前 half/bfloat16 场景不触发实际错误；若未来扩展支持 int8 等 1 字节类型则会触发。
- **测试方案**: 扩展类型支持时进行边界测试，验证不同数据类型下地址偏移正确性。

---

## 汇总表

| 编号 | 位置 | Bug 类型 | 严重程度 | 简要描述 |
|------|------|----------|----------|----------|
| 1 | 第 150 行 (SwishNegOne DAG) | 类型错误 | 严重 | CopyOut 硬编码 float，应为模板类型 T |
| 2 | 第 54/68/107/123 行 | DAG 逻辑错误 | 严重 | Mask 未按剩余元素数更新，多迭代时最后一轮越界 |
| 3 | 第 69/75/124/132 行 | 潜在逻辑风险 | 中等 | vlSize 基于 float 计算但用于 T* 指针偏移，扩展类型时有风险 |

## 修复建议

**Bug 1 修复** (第 150 行):
```cpp
// 修改前
using OpCopyOut = Bind<Vec::CopyOut<float>, Placeholder::Out0<float>, OpResult>;
// 修改后
using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;
```

**Bug 2 修复** (所有 UpdateMask 调用处):
```cpp
// 修改前
mask = AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(count);
// 修改后
uint32_t remaining = count - loopIdx * vlSize;
mask = AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(remaining);
```
