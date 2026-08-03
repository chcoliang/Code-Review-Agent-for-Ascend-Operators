# BatchMatMulV3 Base Tiling 代码审查报告

## Bug 列表

### Bug 1: depthB1 计算使用了错误的数据类型大小

- **位置**: 第 462 行, `DoCommonTiling()` 函数
- **类型**: 数据类型混用错误
- **严重程度**: 高
- **描述**: 计算 B 矩阵在 L1 中的深度 `depthB1` 时，使用了 `aDtypeSize_`（A 矩阵的数据类型大小）而非 `bDtypeSize_`（B 矩阵的数据类型大小）。当 A 和 B 矩阵数据类型不同时（例如 A 为 fp32、B 为 fp16），会导致 B 矩阵的 L1 深度计算错误，可能导致 L1 buffer 溢出或利用率不足。
  ```cpp
  // 错误代码：
  uint64_t depthB1 = (totalL1Size / NUM_TWO / aDtypeSize_ / (baseN * baseK) / 4UL) * 4UL * 2UL;
  // 应为：
  uint64_t depthB1 = (totalL1Size / NUM_TWO / bDtypeSize_ / (baseN * baseK) / 4UL) * 4UL * 2UL;
  ```
- **触发条件**: A 矩阵和 B 矩阵数据类型不同时触发（如 A=float32, B=float16）
- **测试方案**: 构造 A 为 float32、B 为 float16 的 BMM 算子，检查 depthB1 计算值是否与 L1 实际可容纳的 B 块数一致

---

### Bug 2: DoMultiBatchTiling 中 L1 容量计算对 B 矩阵使用了错误的 dtype

- **位置**: 第 590 行, `DoMultiBatchTiling()` 函数
- **类型**: 数据类型混用错误
- **严重程度**: 高
- **描述**: 计算 `iterBatch`（L1 可容纳的 batch 数）时，B 矩阵部分 `shapeK * shapeN` 乘以 `aDtypeSize_` 而非 `bDtypeSize_`。当类型不同时，iterBatch 估算不准确，可能导致 L1 内存越界或多 batch 策略选择失误。
  ```cpp
  // 错误代码：
  uint64_t iterBatch = ops::FloorDiv(compileInfo_.l1Size,
      ((shapeM * shapeK + shapeK * shapeN) * aDtypeSize_ + biasSize));
  // 应为：
  uint64_t iterBatch = ops::FloorDiv(compileInfo_.l1Size,
      (shapeM * shapeK * aDtypeSize_ + shapeK * shapeN * bDtypeSize_ + biasSize));
  ```
- **触发条件**: A、B 数据类型不同，且进入 DoMultiBatchTiling 的多 batch 路径
- **测试方案**: 构造异构 dtype 的大 batch BMM case，验证 iterBatch 是否正确反映 L1 容量

---

### Bug 3: DoMultiBatchL1FullLoadTiling 中 L1 容量计算对 B 矩阵使用了错误的 dtype

- **位置**: 第 705 行, `DoMultiBatchL1FullLoadTiling()` 函数
- **类型**: 数据类型混用错误
- **严重程度**: 高
- **描述**: 与 Bug 2 相同模式。`bBatch` 的计算使用 `aDtypeSize_` 估算 A+B 的 L1 占用，B 矩阵部分 dtype 错误。
  ```cpp
  // 错误代码：
  uint64_t bBatch = ops::FloorDiv(compileInfo_.l1Size,
      ((shapeM * shapeK + shapeK * shapeN) * aDtypeSize_ + biasSize));
  ```
- **触发条件**: A、B dtype 不同，进入 L1FullLoad 多 batch 路径
- **测试方案**: 同 Bug 2

---

### Bug 4: aBatch 计算中 B 矩阵 L1 占用使用了错误的 dtype

- **位置**: 第 709 行, `DoMultiBatchL1FullLoadTiling()` 函数
- **类型**: 数据类型混用错误
- **严重程度**: 高
- **描述**: 计算 L1 中减去 B 矩阵占用后剩余空间可装入的 A batch 数时，B 矩阵尺寸 `(shapeK * shapeN) * aDtypeSize_` 使用了 A 的 dtype size。
  ```cpp
  // 错误代码：
  uint64_t aBatch = ops::FloorDiv(
      compileInfo_.l1Size - ((shapeK * shapeN) * aDtypeSize_ + biasSize) * bBatch,
      (shapeM * shapeK) * aDtypeSize_);
  // B 部分应为 bDtypeSize_：
  uint64_t aBatch = ops::FloorDiv(
      compileInfo_.l1Size - ((shapeK * shapeN) * bDtypeSize_ + biasSize) * bBatch,
      (shapeM * shapeK) * aDtypeSize_);
  ```
- **触发条件**: A、B dtype 不同，进入多 batch L1 全载路径且 bBatch == 1
- **测试方案**: 构造异构 dtype case 且 B 矩阵恰好能放入 L1 一份，验证 aBatch 计算是否溢出

---

### Bug 5: DoMultiBatchL1FullLoadTilingImpl 中 totalL1Size 为死代码且 depthB1 缺少 L1 约束

- **位置**: 第 666-676 行, `DoMultiBatchL1FullLoadTilingImpl()` 函数
- **类型**: 逻辑缺陷 / 死代码
- **严重程度**: 中
- **描述**: 函数计算了 `totalL1Size` 但从未使用。`depthB1` 的计算仅基于 B 矩阵的总元素数与 base block 的比值，没有与 L1 可用空间做约束检查。当 B 矩阵实际大小超出 L1 可用空间时，depthB1 可能过大导致 L1 溢出。
  ```cpp
  uint64_t totalL1Size = compileInfo_.l1Size + reserveSize; // 计算了但从未使用!
  // ...
  uint64_t depthB1 = (shapeN * shapeK * bmmTilingData_.multiBatchInfo.bBatch / (baseN * baseK) / 4) * 4;
  // 缺少: depthB1 = std::min(depthB1, maxDepthFromL1);
  ```
- **触发条件**: 进入 MultiBatchL1FullLoad 路径，且 B 矩阵尺寸较大（bBatch > 1 时放大）
- **测试方案**: 构造 bBatch > 1 且 B 矩阵总大小接近或超过 L1 的 case，检查是否出现 L1 越界

---

### Bug 6: CheckBMMTilingDataIsVaild 函数名拼写错误

- **位置**: 第 200 行
- **类型**: 代码规范 / 拼写错误
- **严重程度**: 低
- **描述**: 函数名 `CheckBMMTilingDataIsVaild` 中 "Vaild" 应为 "Valid"。此为拼写错误，不影响功能但影响代码可维护性。
- **触发条件**: N/A（不影响运行时行为）
- **测试方案**: N/A

---

### Bug 7: PostTiling 中可能重复调用 CalculateNd2nzWorkspaceSize 导致 workspace 计算不一致

- **位置**: 第 1013-1015 行, `PostTiling()` 函数
- **类型**: 逻辑缺陷
- **严重程度**: 中
- **描述**: 在 `DoLibApiTiling()` 的 `DoMultiBatchTiling()` 路径中已调用过 `CalculateNd2nzWorkspaceSize()`（第 627 行），且可能在该函数中对 `batchTileBlock` 做了调整。`PostTiling()` 在条件匹配时重新调用 `CalculateNd2nzWorkspaceSize()`，会将 `workspaceSize_` 重置为 0 后重新计算，但此时 `batchTileBlock` 可能已在首次调用中被修改，导致第二次计算使用的 batchTileBlock 与预期不一致，可能产生 workspace 大小偏差。
  ```cpp
  // PostTiling 中：
  CalculateNd2nzWorkspaceSize();  // 重新计算，workspaceSize_ 被重置
  workspaceSize_ += RPC_WORKSIZE * MB_SIZE;  // 追加 RPC workspace
  ```
- **触发条件**: 进入 MultiBatch + MixNd2Nz 路径，且首次 CalculateNd2nzWorkspaceSize 修改了 batchTileBlock
- **测试方案**: 构造大 workspace 场景（超 L2），验证最终 workspace 大小是否正确

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| Bug 1 | L462 | dtype 混用 | 高 | depthB1 计算用 aDtypeSize_ 替代 bDtypeSize_ |
| Bug 2 | L590 | dtype 混用 | 高 | iterBatch L1 容量计算中 B 部分用错 dtype |
| Bug 3 | L705 | dtype 混用 | 高 | bBatch L1 容量计算中 B 部分用错 dtype |
| Bug 4 | L709 | dtype 混用 | 高 | aBatch 剩余 L1 计算中 B 占用用错 dtype |
| Bug 5 | L666-676 | 逻辑缺陷 | 中 | totalL1Size 死代码，depthB1 缺少 L1 约束 |
| Bug 6 | L200 | 拼写错误 | 低 | 函数名 "Vaild" 应为 "Valid" |
| Bug 7 | L1013-1015 | 逻辑缺陷 | 中 | workspace 重复计算可能导致大小偏差 |

## 总结

本文件最核心的系统性问题是 **B 矩阵相关的 L1 内存容量计算统一错误地使用了 `aDtypeSize_` 而非 `bDtypeSize_`**（Bug 1-4）。这在 A、B 同类型时不会暴露，但当算子支持异构 dtype 输入时将产生严重的内存越界或性能退化。建议全局搜索 `aDtypeSize_` 出现在 B 矩阵计算上下文中的所有位置进行修复。
