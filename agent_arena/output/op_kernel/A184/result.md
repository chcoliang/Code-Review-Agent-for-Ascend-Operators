`MulXfp16Op`输出Cast使用`CAST_MODE_NONE`而非`CAST_MODE_RINT`（第393行），导致float结果转回fp16/bf16时不进行四舍五入而是直接截断，造成精度损失。触发条件：fp16或bf16同类型Mul运算。
