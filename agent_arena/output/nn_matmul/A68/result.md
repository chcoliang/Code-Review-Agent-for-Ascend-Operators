**Bug**: 第219行 `CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_SUCCESS)` 使用了错误的错误码 `ACLNN_SUCCESS`，当空指针检查失败时返回成功状态而非错误，导致后续空指针解引用。触发条件：传入 `self=nullptr` 或 `mat2=nullptr`。
