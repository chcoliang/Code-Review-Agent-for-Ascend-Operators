`MulOp`的`MemCfg`使用`MemLevel::LEVEL_0`而非`MemLevel::LEVEL_2`（第380行），内存优化级别过低导致无法利用多级buffer流水线，严重影响性能，且可能因buffer管理策略不匹配导致运行错误。触发条件：float/int32/int64/int16/complex64类型Mul运算。
