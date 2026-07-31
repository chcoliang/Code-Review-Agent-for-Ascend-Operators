# A63 审查结果: tiling中L1空间计算错误

## Bug

| 项目 | 内容 |
|------|------|
| **文件** | `batch_mat_mul_v3_base_tiling.cpp` |
| **位置** | 第456行 (`DoCommonTiling` 函数), 第666行 (`DoMultiBatchL1FullLoadTilingImpl` 函数), 第944行 (`DoL1FullLoadTiling` 函数) |
| **问题代码** | `uint64_t totalL1Size = compileInfo_.l1Size + reserveSize;` (第456行、第666行) 以及 `const uint64_t totalL1Size = compileInfo_.l1Size + 256;` (第944行) |
| **描述** | L1可用空间计算使用了加法 (`+`) 而非减法 (`-`)。注释明确说明 "256B为预留给rpc使用"，即这256B应从L1总空间中预留出去（减去），而非额外增加。使用加法会导致 `totalL1Size` 比实际可用的L1空间大256B，后续基于 `totalL1Size` 计算的 `depthA1`、`depthB1`、`stepKa`、`stepKb` 等tiling参数可能偏大，分配的数据块超出L1实际容量。 |
| **触发条件** | 当矩阵规模使得L1空间接近满载时（即 `baseM * baseK * aDtypeSize_` 或 `baseN * baseK * bDtypeSize_` 接近 L1 容量边界），多出的256B会导致计算出的depth值恰好多一级，使实际搬运数据量超出L1物理空间。 |
| **预期异常** | L1 buffer溢出，写入L1的数据覆盖相邻buffer区域，导致Cube计算读取到脏数据，输出结果出现随机精度错误；严重情况下可能触发硬件内存保护异常（bus error）或产生不可预测的计算行为。在极端场景下可能导致NPU hang。 |
