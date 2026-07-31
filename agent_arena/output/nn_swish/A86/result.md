**Bug:** 第59行 `CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_SUCCESS)` 错误码应为 `ACLNN_ERR_PARAM_NULLPTR` 却写成 `ACLNN_SUCCESS`。当 self/out 为空时，CHECK_RET 判断返回值!=ACLNN_SUCCESS 不成立，空指针检查形同虚设，后续访问空指针导致崩溃。

**触发输入:** 传入 self=nullptr 或 out=nullptr 调用 aclnnSwishGetWorkspaceSize。
