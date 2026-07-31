`DCACHE_SIZE`被错误设为`32 * 1024 * 4`（128KB）而非正确的`32 * 1024`（32KB）（第33行），导致PostTiling中SetLocalMemorySize大幅缩减可用UB空间，且double分支extraSize过大，tiling计算错误甚至可能导致UB溢出。触发条件：任何Mul算子执行。
