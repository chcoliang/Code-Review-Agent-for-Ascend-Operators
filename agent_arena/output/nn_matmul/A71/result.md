**Bug**: 第116-117行 `CheckDtypeValid` 中缺少对 `out` 张量的dtype校验（`OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeList, return false)` 被删除），导致输出张量dtype不合法时未被拦截。触发条件：out张量dtype为不支持的类型如DT_INT32。
