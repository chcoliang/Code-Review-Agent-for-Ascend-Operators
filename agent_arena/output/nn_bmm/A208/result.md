# BatchMatMulV3 Base Tiling 代码审查报告

## Bug 列表

### Bug 1: 常量 ALIGNMENT_32 值与命名不一致

- **位置**: 第 40 行
- **类型**: 常量定义错误
- **严重程度**: 高
- **描述**: `constexpr uint64_t ALIGNMENT_32 = 128;` 常量名为 `ALIGNMENT_32`，语义暗示32字节对齐，但实际值为128。如果意图是32字节对齐，值应为32；如果意图是128字节对齐，名称应为 `ALIGNMENT_128`。此错误会导致所有依赖该常量进行对齐计算的位置产生错误的对齐粒度（比预期大4倍或名称误导开发者）。
- **触发条件**: 任何使用 `ALIGNMENT_32` 进行对齐计算的代码路径（可能在头文件或其他编译单元中引用）。
- **测试方案**: 检查所有引用 `ALIGNMENT_32` 的代码，验证其对齐逻辑是否需要32字节还是128字节；构造需要精确32B对齐的tensor输入，观察是否产生多余padding或地址不对齐错误。

---

### Bug 2: depthB1 计算误用 aDtypeSize_ 代替 bDtypeSize_

- **位置**: 第 462 行，`DoCommonTiling()` 函数
- **类型**: 变量引用错误（逻辑Bug）
- **严重程度**: 高
- **描述**: `uint64_t depthB1 = (totalL1Size / NUM_TWO / aDtypeSize_ / (baseN * baseK) / 4UL) * 4UL;` 计算矩阵B在L1中的深度时，错误地使用了矩阵A的数据类型大小 `aDtypeSize_`，而非矩阵B的 `bDtypeSize_`。当A/B数据类型不同（例如混合精度场景：A为fp16=2字节，B为fp32=4字节）时，depthB1被高估一倍，导致实际L1内存使用超出容量，可能引起数据覆盖或硬件异常。
- **触发条件**: A矩阵和B矩阵数据类型大小不同的场景，如 A=fp16(2B), B=fp32(4B)。
- **测试方案**: 构造混合精度BMM算子（A为fp16，B为fp32），设置较大的K和N使L1接近满载，验证计算结果正确性；检查是否有L1越界访问。

---

### Bug 3: totalL1Size 计算方向错误（加法应为减法）

- **位置**: 第 456 行、第 666 行、第 944 行
- **类型**: 算术逻辑错误
- **严重程度**: 高
- **描述**: `uint64_t totalL1Size = compileInfo_.l1Size + reserveSize;` 注释说明 "256B为预留给rpc使用，单算子不涉及"，含义是L1总空间中有256B需要预留给RPC。对于可用空间计算，应该是减去预留部分：`compileInfo_.l1Size - reserveSize`。当前写法使可用L1空间比实际多了512字节（多加了256而非减256），可能导致L1使用量计算略微超出物理限制。此bug在三处出现（DoCommonTiling、DoMultiBatchL1FullLoadTilingImpl、DoL1FullLoadTiling）。
- **触发条件**: 当矩阵规模使L1使用量刚好接近L1物理容量边界时，多出的512B可能导致越界。
- **测试方案**: 构造使L1用量接近满载的shape（如大K值），对比加/减256时tiling参数差异；在硬件上运行验证是否出现L1 overflow。

---

### Bug 4: CheckBMMTilingDataIsVaild 函数名拼写错误

- **位置**: 第 200 行（声明）、第 316 行（调用）
- **类型**: 命名规范问题（低风险）
- **严重程度**: 低
- **描述**: 函数名 `CheckBMMTilingDataIsVaild` 中 "Vaild" 应为 "Valid"。虽不影响功能，但影响代码可维护性和可搜索性。
- **触发条件**: N/A（不影响运行时行为）。
- **测试方案**: 全局搜索修正拼写，确保所有调用点同步更新。

---

### Bug 5: DoMultiBatchL1FullLoadTilingImpl 中 depthB1 计算未使用 totalL1Size

- **位置**: 第 676 行，`DoMultiBatchL1FullLoadTilingImpl()` 函数
- **类型**: 逻辑错误
- **严重程度**: 中
- **描述**: `uint64_t depthB1 = (shapeN * shapeK * bmmTilingData_.multiBatchInfo.bBatch / (baseN * baseK) / 4) * 4;` 此处计算BL1的depth时，使用的是矩阵B的完整shape元素数除以base块大小，但没有考虑L1容量限制（未使用 `totalL1Size`）。当B矩阵较大时，计算出的depthB1可能超出L1实际可用空间。对比 `DoCommonTiling()` 中的计算方式 `(totalL1Size / NUM_TWO / dtypeSize / (baseN * baseK) / 4UL) * 4UL`，此处缺少L1容量约束。
- **触发条件**: B矩阵shape较大且bBatch > 1时，depthB1超出L1物理容量。
- **测试方案**: 构造大shape的B矩阵（如N=1024, K=1024, bBatch=4），检查depthB1是否超出L1容量限制。

---

### Bug 6: PostTiling 中可能重复调用 CalculateNd2nzWorkspaceSize

- **位置**: 第 1014 行，`PostTiling()` 函数
- **类型**: 逻辑冗余/潜在计算错误
- **严重程度**: 中
- **描述**: 在 `PostTiling()` 中，当条件满足 MultiBatch+MixNd2Nz 时，会再次调用 `CalculateNd2nzWorkspaceSize()`。但在 `DoMultiBatchTiling()` 的第 627 行已经调用过一次。重复调用会重置 `workspaceSize_` 为0（第769行），然后重新计算。如果在两次调用之间 `workspaceSize_` 已被其他逻辑修改（如加上 RPC_WORKSIZE），则第二次调用会丢失之前累加的值。此外第1015行 `workspaceSize_ += RPC_WORKSIZE * MB_SIZE` 只在重复调用后执行，逻辑上RPC workspace可能被遗漏或重复。
- **触发条件**: MultiBatch + MixNd2Nz 路径下进入PostTiling。
- **测试方案**: 跟踪 workspaceSize_ 在整个tiling流程中的变化，验证最终workspace大小是否正确包含nd2nz和RPC空间。

---

## 汇总表

| 编号 | 位置 | 类型 | 严重程度 | 简要描述 |
|------|------|------|----------|----------|
| 1 | 第40行 | 常量定义错误 | 高 | ALIGNMENT_32 值为128，名称与值矛盾 |
| 2 | 第462行 | 变量引用错误 | 高 | depthB1计算使用aDtypeSize_而非bDtypeSize_ |
| 3 | 第456/666/944行 | 算术逻辑错误 | 高 | L1可用空间应减去预留值，实际却加上 |
| 4 | 第200行 | 命名拼写错误 | 低 | "Vaild" 应为 "Valid" |
| 5 | 第676行 | 逻辑错误 | 中 | depthB1计算缺少L1容量约束 |
| 6 | 第1014行 | 逻辑冗余 | 中 | CalculateNd2nzWorkspaceSize可能被重复调用导致workspace计算异常 |
