`NANI`常量被错误设为`0x00000000`而非`0x7FF80000`（第37行），导致double乘法中NaN输入输出的结果为0而非NaN，违反IEEE 754标准。触发条件：double类型Mul运算中任一输入为NaN。
