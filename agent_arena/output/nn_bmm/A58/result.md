# A58 BatchMatMul Code Review

## Bug: UB_SIZE 常量值错误（24KB 应为 248KB）

| 项目 | 内容 |
|------|------|
| **位置** | 匿名命名空间常量定义, 第 78 行: `static const uint64_t UB_SIZE = 24UL * 1024UL;` |
| **描述** | UB_SIZE 被错误设置为 `24UL * 1024UL`（24KB），正确值应为 `248UL * 1024UL`（248KB）。Ascend 910B 的 Unified Buffer 大小为 248KB。该常量在 `CheckShapeEqualToMul` 函数中用于判断 M、N 对齐后的数据量是否能放入 UB：`(alignM + alignN + alignM * alignN) * dataSize > UB_SIZE`。由于阈值缩小了约 10 倍，大量本应走 Mul 优化路径（K=1 场景下的 element-wise 乘法替代）的 shape 会被错误拒绝，转而走 Cube BatchMatMul 路径。虽然功能结果正确，但会导致严重的性能退化（Cube 路径在 K=1 小 shape 场景下效率远低于 Vector Mul）。 |
| **触发输入** | `self`: dtype=DT_FLOAT16, shape=[256,4,1]; `mat2`: dtype=DT_FLOAT16, shape=[256,1,32]; `out`: dtype=DT_FLOAT16, shape=[256,4,32]; `cubeMathType`=0。此时 K=1，M=4，N=32，alignM=32，alignN=32，(32+32+32*32)*2=2176 字节 < 248KB 但 > 24KB。 |
| **预期异常** | 应走入 Mul 优化路径（K=1 + shape 满足 UB 容量），实际因 UB_SIZE 过小而误判 UB 溢出，被迫走 Cube BatchMatMul 路径，导致计算性能严重退化（非功能性 bug，但属于 Cube 特有的 UB 资源判断错误）。 |
