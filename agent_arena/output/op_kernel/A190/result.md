`MulOp`中对InputX1插入了冗余的同类型Cast `Vec::Cast<T, T, CAST_MODE_NONE>`（第377行），对于int64_t等不支持同类型Cast的类型会导致编译失败或运行时错误，且额外占用一个UB buffer。触发条件：int64_t/int16_t等类型的Mul运算。
