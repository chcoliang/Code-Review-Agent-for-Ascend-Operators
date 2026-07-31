`MulOp`跳过了`Vec::CopyInBrc`直接使用GM Placeholder作为Mul输入（第373-374行），数据未从GM搬运到UB就执行向量计算，导致操作未定义内存/垃圾数据，计算结果完全错误。触发条件：float/int32/int64/int16/complex64类型Mul运算。
