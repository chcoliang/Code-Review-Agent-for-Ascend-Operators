**Bug**: 第56-57行 `DTYPE_SUPPORT_LIST` 缺少 `DataType::DT_BF16`，仅包含 `{DT_FLOAT, DT_FLOAT16}`，导致在支持BF16的910B平台上BF16输入被错误拒绝。触发条件：在Ascend 910B上输入BF16类型的self和mat2张量。
