`CheckMulShape`函数跳过了广播shape推导，直接使用`self->GetViewShape()`作为目标shape（第296行），导致self和other shape不同时shape校验错误，合法的broadcast输入会被拒绝或非法输入被放行。触发条件：self和other shape不同但可broadcast时（如[3,1]和[1,4]）。
