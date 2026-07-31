**Bug:** 第58-68行 `CheckParams` 中缺少 `CheckDtypeValidBetaToFloat(betaOptional)` 调用，beta 参数的数据类型未被校验。当传入不可转换为 float 的 beta 类型时，后续 `betaOptional->ToFloat()` 可能产生错误值或未定义行为。

**触发输入:** 传入 dtype 为复数类型（如 DT_COMPLEX64）的 betaOptional 标量参数。
