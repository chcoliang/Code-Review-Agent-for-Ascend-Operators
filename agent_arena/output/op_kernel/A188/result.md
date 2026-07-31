`MulOp`的`MemCfg`使用`MemLevel::LEVEL_1`而非`MemLevel::LEVEL_2`（第380行），内存优化级别错误导致float/int32/int64/int16/complex64类型的buffer管理策略不正确，可能造成UB内存分配异常或性能严重下降。触发条件：float/int32/int64/int16/complex64类型Mul运算。
