# Mul Tiling 代码审查报告

审查文件：`mul_tiling_arch35.cpp`
目标平台：Ascend 910B, CANN 8.5.0

---

## Bug #1: SetLocalMemorySize 参数计算错误（加法应为减法）

- **位置**：第 210 行，`PostTiling()` 函数
- **类型**：内存大小计算逻辑错误
- **严重程度**：严重（Critical）

**描述**：

```cpp
context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ + DCACHE_SIZE));
```

`SetLocalMemorySize` 用于设置 kernel 可使用的 UB 内存大小。`ubSize_` 已经是从平台获取的完整 UB 硬件容量（910B 上通常为 192KB 或 256KB）。此处将 `DCACHE_SIZE`（32KB）**加到** UB 总大小上，导致设置的本地内存大小**超过实际硬件 UB 容量**。

正确逻辑应为 `ubSize_ - DCACHE_SIZE`，即从 UB 中预留 DCACHE 区域（32KB）给系统/硬件使用，剩余部分供算子计算使用。符号方向错误（`+` 应为 `-`）。

**触发条件**：所有 dtype 组合的 Mul 算子执行均会触发。

**预期异常**：kernel 执行时实际使用的内存地址超出 UB 物理边界，导致内存越界访问（OOB），可能引发硬件异常、数据错误或 AICORE 挂死。

---

## Bug #2: DCACHE_SIZE 在 PostTiling 中对所有分支无条件生效

- **位置**：第 210 行（`PostTiling`）与第 147 行（double 分支 DoTiling 调用）的逻辑不一致
- **类型**：逻辑一致性错误
- **严重程度**：中等（Medium）

**描述**：

在 `DoTiling` 调用中，只有 double 分支传递了 `extraSize = DCACHE_SIZE`：

```cpp
// 第 147 行 - 仅 double 分支
DoTiling<MulDoubleOp<double>::OpDag>(tiling->GetContext(), tiling->tilingKey_, DCACHE_SIZE, 0);
```

其他所有分支（int8, fp16, float, int32 等）调用 `DoTiling` 时 `extraSize` 默认为 0。

然而，`PostTiling()`（第 210 行）对**所有分支**无条件地将 DCACHE_SIZE 加入（或应减去）到 LocalMemorySize 中。这意味着：

- 非 double 分支：tiling 计算时未考虑 DCACHE 额外空间，但 PostTiling 却修改了内存大小声明，导致 tiling 切分策略与实际可用内存不匹配。
- 对于非 double 分支，如果意图是预留 DCACHE（即应减去），则 tiling 计算基于完整 `ubSize_` 进行切分，但实际可用内存少了 32KB，可能导致 buffer 溢出。

**触发条件**：使用非 double 数据类型（如 float、fp16、int32 等）的 Mul 算子执行时触发。

**预期异常**：tiling 切分策略与实际可用 UB 空间不匹配。若实际可用空间小于 tiling 计算假设的空间，kernel 运行时可能发生 UB 内存溢出，导致计算结果错误或硬件异常。

---

## 审查总结

| 编号 | 位置 | 类型 | 严重程度 |
|------|------|------|----------|
| #1 | 第 210 行 PostTiling | 内存大小计算符号错误（+ 应为 -） | Critical |
| #2 | 第 210 行 vs 第 147 行 | DCACHE_SIZE 应用范围与 tiling 计算不一致 | Medium |
