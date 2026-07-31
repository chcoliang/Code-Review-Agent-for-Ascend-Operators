**Bug:** 第94-96行缺少空张量提前返回逻辑（`if (self->IsEmpty() || out->IsEmpty()) { ... return ACLNN_SUCCESS; }`），空张量会继续进入 Contiguous/Swish/ViewCopy 等计算流程，对零元素张量执行计算可能导致未定义行为或资源浪费。

**触发输入:** 传入 shape 包含 0 维度的空张量（如 shape=[0] 或 shape=[3,0,4]）。
