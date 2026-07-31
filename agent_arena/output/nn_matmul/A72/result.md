**Bug**: 第893-894行 `aclnnMatmulGetWorkspaceSize` 中删除了空tensor提前返回逻辑（`if (out->IsEmpty() && (self->IsEmpty() || mat2->IsEmpty()))` 分支），导致空tensor场景下仍尝试构建计算图和执行，可能返回错误。触发条件：self和out都是空tensor（某维为0）。
