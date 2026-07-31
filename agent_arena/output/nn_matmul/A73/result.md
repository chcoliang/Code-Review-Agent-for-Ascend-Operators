**Bug**: 第566-567行 `MatMulDotGraph::PreProcess` 中对 `self` 的 `l0op::Contiguous` 调用被删除（仅保留null检查），导致非连续内存布局的1D张量在dot运算时使用原始非连续数据，产生计算结果错误。触发条件：输入非连续的1D张量执行dot运算（如经过slice的张量）。
