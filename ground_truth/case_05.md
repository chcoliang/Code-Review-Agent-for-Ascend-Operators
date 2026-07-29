# Ground Truth: case_05_dtype_whitelist

## Bug 信息

| 项目 | 内容 |
|------|------|
| **文件** | `math/mul/op_api/aclnn_mul.cpp` |
| **函数** | `ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST` |
| **行号** | 第 58 行 |
| **错误分类** | 一类·参数校验 (1.2) — dtype白名单遗漏 |
| **注入方式** | 从支持列表中删除 `DataType::DT_DOUBLE` |

## Bug 描述

`ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST` 中遗漏了 `DT_DOUBLE`，导致合法的 DOUBLE 类型输入在 API 参数校验阶段被错误拒绝，返回 dtype 不支持的错误码。

## 正确代码

```cpp
static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_DOUBLE, DataType::DT_INT8,
  DataType::DT_UINT8, DataType::DT_INT16, DataType::DT_INT64, DataType::DT_BOOL, DataType::DT_COMPLEX128,
  DataType::DT_COMPLEX64, DataType::DT_BF16};
```

## 触发方式

构造 `self`、`other`、`out` 均为 `dtype=DT_DOUBLE, shape=[2,2]` 的合法 tensor。调用 `aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor)`。

## 预期结果

- 正确算子：返回 `ACLNN_SUCCESS`
- Buggy 算子：返回 dtype 不支持错误码 (非 SUCCESS)

## 评分建议

| 维度 | 2 分标准 |
|------|------|
| bug 发现 | 定位到 `ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST` 第 58 行，指出 `DT_DOUBLE` 缺失 |
| 测试生成 | 测试代码使用 `DT_DOUBLE` 类型 tensor 调用 `aclnnMulGetWorkspaceSize` |
| 解释修复 | 恢复 `DataType::DT_DOUBLE` 到支持列表 |
