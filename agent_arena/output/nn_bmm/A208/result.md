# BatchMatMulV3 Base Tiling 代码审查报告

## Bug 1: 常量 ALIGNMENT_32 值与命名不一致

- **位置**: 第 40 行
- **类型**: 常量定义错误
- **严重程度**: 高
- **描述**: 常量命名为 `ALIGNMENT_32`，语义上应表示 32 字节对齐，但实际赋值为 128。这会导致所有使用该常量进行对齐计算的地方产生错误的对齐结果（对齐到 128 而非预期的 32）。
- **代码**:
  ```cpp
  constexpr uint64_t ALIGNMENT_32 = 128;  // 应为 32
  ```
- **触发条件**: 任何依赖 `ALIGNMENT_32` 进行 32 字节对齐计算的路径均会触发。
- **修复建议**: 将值修改为 `32`，或将常量名修改为 `ALIGNMENT_128`（取决于实际设计意图）。
- **测试方案**: 检查所有引用 `ALIGNMENT_32` 的代码路径，构造需要 32B 对齐的输入 shape，验证对齐计算结果是否正确。

---

## Bug 2: L1 可用空间计算方向错误（加法应为减法）

- **位置**: 第 456 行、第 666 行、第 944 行
- **类型**: Tiling 参数计算错误
- **严重程度**: 高
- **描述**: 计算 L1 可用总空间时，代码将预留空间（reserveSize = 256B）**加到** L1 总大小上，但注释明确写着"256B为预留给rpc使用"，即应从总空间中**减去**预留部分。这导致计算出的可用 L1 空间比实际大 512 字节（多加了 256 而非减去 256），可能导致 L1 buffer 溢出。
- **代码**:
  ```cpp
  // 第 456 行 (DoCommonTiling)
  uint64_t totalL1Size = compileInfo_.l1Size + reserveSize;
  // 第 666 行 (DoMultiBatchL1FullLoadTilingImpl)
  uint64_t totalL1Size = compileInfo_.l1Size + reserveSize;
  // 第 944 行 (DoL1FullLoadTiling)
  const uint64_t totalL1Size = compileInfo_.l1Size + 256;
  ```
- **触发条件**: 所有进入 DoCommonTiling、DoMultiBatchL1FullLoadTilingImpl 或 DoL1FullLoadTiling 的执行路径。当 L1 实际占用接近上限时，多出的 512 字节会导致越界访问。
- **修复建议**: 将 `+` 改为 `-`：
  ```cpp
  uint64_t totalL1Size = compileInfo_.l1Size - reserveSize;
  ```
- **测试方案**: 构造使 L1 占用接近满载的大 shape（如 M=2048, K=2048, FP32），检查是否出现 L1 内存越界或数据覆写。

---

## Bug 3: B 矩阵 depth 计算误用 A 矩阵数据类型大小

- **位置**: 第 462 行
- **类型**: Tiling 参数计算错误
- **严重程度**: 高
- **描述**: 计算 `depthB1`（B 矩阵在 L1 中的深度）时，使用了 `aDtypeSize_`（A 矩阵的数据类型大小）而非 `bDtypeSize_`（B 矩阵的数据类型大小）。当 A 和 B 的数据类型不同时（例如混合精度场景），会导致 B 矩阵的 L1 空间分配计算错误。
- **代码**:
  ```cpp
  uint64_t depthB1 = (totalL1Size / NUM_TWO / aDtypeSize_ / (baseN * baseK) / 4UL) * 4UL;
  //                                          ^^^^^^^^^^^  应为 bDtypeSize_
  ```
- **触发条件**: 当 A 矩阵和 B 矩阵数据类型不同时触发（例如 A 为 FP16、B 为 FP32，或混合精度推理场景）。若 `aDtypeSize_ < bDtypeSize_`，depthB1 会偏大，导致 L1 溢出；反之则浪费 L1 空间。
- **修复建议**: 将 `aDtypeSize_` 替换为 `bDtypeSize_`：
  ```cpp
  uint64_t depthB1 = (totalL1Size / NUM_TWO / bDtypeSize_ / (baseN * baseK) / 4UL) * 4UL;
  ```
- **测试方案**: 构造 A 为 FP16、B 为 FP32 的混合精度 BMM 输入，验证 depthB1 计算是否正确，以及是否出现 L1 溢出或计算结果错误。

---

## Bug 4: DoMultiBatchL1FullLoadTilingImpl 中 totalL1Size 未被使用

- **位置**: 第 665-670 行
- **类型**: 逻辑缺陷
- **严重程度**: 低
- **描述**: `DoMultiBatchL1FullLoadTilingImpl` 函数中计算了 `totalL1Size` 并做了 bias 相关调整，但该变量后续未参与 `depthB1` 的计算（第 676 行使用的是 shape 面积除以 base 面积的方式），属于死代码或遗漏使用。
- **触发条件**: 进入 DoMultiBatchL1FullLoadTilingImpl 路径。
- **修复建议**: 确认 depthB1 计算是否应受 totalL1Size 约束，如需约束则加入 L1 容量限制逻辑。
- **测试方案**: 构造使 depthB1 计算结果超过 L1 实际容量的 case，验证是否溢出。

---

## Bug 5: CheckBMMTilingDataIsVaild 函数名拼写错误

- **位置**: 第 200 行
- **类型**: 代码规范/拼写错误
- **严重程度**: 低
- **描述**: 函数名 `CheckBMMTilingDataIsVaild` 中 "Vaild" 应为 "Valid"。虽不影响功能，但会影响代码可维护性和搜索。
- **触发条件**: N/A（不影响运行时行为）
- **修复建议**: 重命名为 `CheckBMMTilingDataIsValid`。
- **测试方案**: 编译确认无引用断裂。

---

# 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 40 | 常量定义错误 | 高 | `ALIGNMENT_32` 值为 128，与命名语义矛盾 |
| 2 | 456, 666, 944 | Tiling参数计算错误 | 高 | L1 可用空间应减去预留值，代码误用加法 |
| 3 | 462 | Tiling参数计算错误 | 高 | B矩阵depth计算误用A矩阵dtype大小 |
| 4 | 665-676 | 逻辑缺陷 | 低 | totalL1Size 计算后未实际约束 depthB1 |
| 5 | 200 | 代码规范 | 低 | 函数名拼写错误 "Vaild" → "Valid" |
