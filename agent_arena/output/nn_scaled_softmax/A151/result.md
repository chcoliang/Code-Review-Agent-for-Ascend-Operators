# Code Review Result - A151

## Bug 1: fixedTriuMask 判断条件取反，逻辑完全相反

- **位置**: 第 132 行
- **类型**: 逻辑错误 / 条件判断取反
- **严重程度**: 高
- **描述**: 在 `aclnnScaledMaskedSoftmaxGetWorkspaceSize` 函数中，对 `fixedTriuMask` 参数的校验条件为 `if (!fixedTriuMask)`，但错误日志明确说明 "the param fixedTriuMask only suppport false"。这意味着当 fixedTriuMask=false（合法值）时反而会报错返回，而当 fixedTriuMask=true（非法值）时反而会被放行继续执行。逻辑完全与设计意图相反。
- **触发条件**:
  - 当用户正确传入 `fixedTriuMask = false` 时，函数会错误地返回 `ACLNN_ERR_PARAM_INVALID`，导致正常功能无法使用。
  - 当用户传入 `fixedTriuMask = true` 时，非法参数不会被拦截，可能导致后端计算行为未定义。
- **修复建议**: 将第 132 行的 `!fixedTriuMask` 改为 `fixedTriuMask`：
  ```cpp
  if (fixedTriuMask) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the param fixedTriuMask only suppport false.");
      return ACLNN_ERR_PARAM_INVALID;
  }
  ```
- **测试方案**:
  1. 构造合法输入，设置 `fixedTriuMask = false`，调用 aclnnScaledMaskedSoftmaxGetWorkspaceSize。
  2. 预期应返回 ACLNN_SUCCESS；若 bug 存在则返回 ACLNN_ERR_PARAM_INVALID。
  3. 设置 `fixedTriuMask = true`，预期应返回 ACLNN_ERR_PARAM_INVALID；若 bug 存在则返回 ACLNN_SUCCESS。

---

## 汇总表

| 编号 | 行号 | 类型 | 严重程度 | 简述 |
|------|------|------|----------|------|
| 1 | 132 | 逻辑错误 | 高 | fixedTriuMask 判断条件取反，合法参数被拒绝，非法参数被放行 |
