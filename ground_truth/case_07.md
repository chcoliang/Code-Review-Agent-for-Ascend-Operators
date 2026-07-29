# Ground Truth: case_07_errorcode_masking

## Bug 信息

| 项目 | 内容 |
|------|------|
| **文件** | `math/mul/op_api/aclnn_mul.cpp` |
| **函数** | `CheckMulParams()` |
| **行号** | 第 322 行 |
| **错误分类** | 一类·参数校验 (1.4) — 错误码伪装 |
| **注入方式** | `ACLNN_ERR_PARAM_NULLPTR` → `ACLNN_SUCCESS` |

## Bug 描述

`CheckMulParams()` 中调用 `CheckMulNotNull()` 后的错误返回码从 `ACLNN_ERR_PARAM_NULLPTR` 改为 `ACLNN_SUCCESS`，导致空指针校验失败时错误被吞噬，后续代码对空指针解引用会导致段错误。

## 正确代码

```cpp
inline static aclnnStatus CheckMulParams(...) {
  CHECK_RET(CheckMulNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);  // ← 被改成了 ACLNN_SUCCESS
  ...
}
```

## 触发方式

构造 `self` 和 `other` 为合法 tensor（`shape=[4,2], dtype=FLOAT`），`out` 传 `nullptr`。调用 `aclnnMulGetWorkspaceSize(self, other, nullptr, &workspaceSize, &executor)`。

## 预期结果

- 正确算子：返回 `ACLNN_ERR_PARAM_NULLPTR (161001)`
- Buggy 算子：段错误 (SEGFAULT, exit=139)

## 评分建议

| 维度 | 2 分标准 |
|------|------|
| bug 发现 | 定位到 `CheckMulParams()` 第 322 行，指出错误码被伪装为 SUCCESS |
| 测试生成 | 测试代码传 `nullptr` 作为 `out` 参数，验证返回码 |
| 解释修复 | 恢复 `ACLNN_ERR_PARAM_NULLPTR` 错误码 |
