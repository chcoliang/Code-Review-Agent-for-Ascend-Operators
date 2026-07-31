**Bug**: 第224-226行 `CheckParam` 中删除了 `CheckMathType(self, mat2, cubeMathType)` 的调用，导致非法的 `cubeMathType` 参数无法被拦截，可能触发未定义的计算行为。触发条件：传入不合法的 `cubeMathType` 值（如超出枚举范围的值）。
