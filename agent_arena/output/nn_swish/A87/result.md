**Bug:** 第63行额外添加了 `if (self->GetDataType() != out->GetDataType()) { return ACLNN_ERR_PARAM_INVALID; }` 严格要求输入输出 dtype 完全相同，这会拒绝框架支持的合法类型提升场景（如 fp16 输入 fp32 输出），与 `CheckDtypeValidActivation` 允许的隐式转换行为冲突。

**触发输入:** 传入 self dtype=FLOAT16、out dtype=FLOAT32 的合法类型提升调用。
