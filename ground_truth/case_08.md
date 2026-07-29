# Ground Truth: case_08_dim_check

## Bug 信息

| 项目 | 内容 |
|------|------|
| **文件** | `math/mul/op_api/aclnn_mul.cpp` |
| **函数** | `CheckMulShape()` |
| **行号** | 第 294 行 |
| **错误分类** | 四类·Shape/广播 (4.2) — 维度上限检查缺失 |
| **注入方式** | 删除两行 `OP_CHECK_MAX_DIM` 调用 |

## Bug 描述

`CheckMulShape()` 中对 `self` 和 `other` 的维度上限检查 `OP_CHECK_MAX_DIM` 被删除，导致超过最大支持维度数的 tensor 通过校验，可能在后续 tiling/kernel 阶段产生内存越界或未定义行为。

## 正确代码

```cpp
inline static bool CheckMulShape(...) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);   // ← 被删除
  OP_CHECK_MAX_DIM(other, MAX_SUPPORT_DIMS_NUMS, return false);  // ← 被删除
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, dstShape, return false);
  ...
}
```

## 触发方式

构造 `self` 为超过 `MAX_SUPPORT_DIMS_NUMS`（通常为 8）维度的 tensor，如 `shape=[1,1,1,1,1,1,1,1,1], dtype=FLOAT`。调用 `aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor)`。

## 预期结果

- 正确算子：返回维度超限错误码
- Buggy 算子：返回 `ACLNN_SUCCESS`（超维 tensor 通过校验）

## 评分建议

| 维度 | 2 分标准 |
|------|------|
| bug 发现 | 定位到 `CheckMulShape()` 第 294 行，指出 `OP_CHECK_MAX_DIM` 缺失 |
| 测试生成 | 测试代码使用超过 8 维的 tensor 调用 `aclnnMulGetWorkspaceSize` |
| 解释修复 | 恢复 `OP_CHECK_MAX_DIM(self, ...)` 和 `OP_CHECK_MAX_DIM(other, ...)` |
