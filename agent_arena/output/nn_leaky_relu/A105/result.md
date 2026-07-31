**Bug**: 第73行添加了 `if (self->GetDataType() != out->GetDataType()) { return ACLNN_ERR_PARAM_INVALID; }` 过度限制检查。LeakyReLU API 允许输入输出 dtype 不同（通过 Cast 转换），此检查会错误拒绝合法的跨类型调用场景。

**触发输入**: 传入 dtype 为 FLOAT16 的 self tensor，out tensor dtype 为 FLOAT32，将被错误拒绝。
