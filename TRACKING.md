# Bug 发现追踪矩阵

> 使用 SKILL_error_injection_eval.md 流程评测。
> Agent 目标：识别 bug + 给出能暴露 bug 的测试输入和验证程序（不要求修复）。
> 验证标准：运行验证代码后输出能看出代码有问题。

## 评测结果

| Case | 注入错误 | Agent检出 | NPU编译 | NPU运行 | 能暴露注入bug | 说明 |
|------|---------|:---:|:---:|:---:|:---:|------|
| A01 | 1.1 out空指针校验删除 | ✅ | ✅ | ✅ exit=0 | ✅ | 正确版返回161001；buggy版SEGFAULT |
| A02 | 4.3 输出Shape校验删除 | ✅ | ✅ | ✅ exit=0 | ✅ | 正确版返回161002；buggy版返回0 |
| A03 | 2.1 DT_DOUBLE被误拒 | ✅ | ✅ | ✅ exit=0 | ✅ | 输出显示DOUBLE被错误拒绝(561103) |
| A04 | 3.2 空Tensor处理删除 | ✅ | ✅ | ⚠️ 超时 | ⚠️ | Agent正确识别，验证代码执行aclnnMul时超时 |
| A05 | 1.2 DT_DOUBLE从白名单删除 | ❌ | ✅ | ✅ exit=0 | ❌ | 发现的是canUseMuls精度问题(非注入) |
| A06 | 1.3 DT_UINT32加入白名单 | ❌ | ✅ | ✅ exit=0 | ❌ | 发现的是workspaceSize空指针(非注入) |
| A07 | 1.4 错误码伪装 | ✅ | ✅ | ✅ exit=0 | ✅ | 正确版返回161001；buggy版返回0+SEGFAULT |
| A08 | 4.2 OP_CHECK_MAX_DIM删除 | ❌ | ✅ | ✅ exit=1 | ❌ | 发现的是workspaceSize空指针(实际触发了真实bug) |

## 汇总

| 指标 | 值 |
|------|:--:|
| 注入 bug 检出率 | **5/8 = 62.5%** |
| 验证代码编译率 | 8/8 = 100% |
| 验证能暴露注入bug | **4/8 = 50%** (A01,A02,A03,A07) |
| 未检出 case | A05(白名单遗漏), A06(白名单过宽), A08(MAX_DIM删除) |

## Agent 各 case 首要发现

| Case | Agent Bug 1 | 是否命中注入 |
|------|------|:---:|
| A01 | CheckMulNotNull 用(void)out忽略out校验 | ✅ |
| A02 | CheckMulShape 用(void)out跳过输出shape验证 | ✅ |
| A03 | 466行硬编码拒绝DT_DOUBLE，与支持列表矛盾 | ✅ |
| A04 | aclnnMulGetWorkspaceSize唯独缺空tensor处理 | ✅ |
| A05 | canUseMuls忽略inferDtype导致FP16溢出 | ❌ |
| A06 | workspaceSize/executor空指针未检查 | ❌ |
| A07 | CheckMulParams第322行错误码ACLNN_SUCCESS | ✅ |
| A08 | workspaceSize空指针解引用 | ❌ |

## NPU 实测输出摘要

```
A01: "Actual: returned status = 161001" → 正确版有防护，buggy版会崩溃
A02: "aclnnMulGetWorkspaceSize returned: 161002" → 正确版拒绝，buggy版放行
A03: "实际返回值: 561103, 正确行为: 应返回 ACLNN_SUCCESS(0)" → 明确暴露bug
A04: "返回状态码: 0, workspaceSize: 0" → 后续执行超时
A05: "Buggy result: inf [OVERFLOW!]" → 暴露的是另一个bug(canUseMuls)
A06: "Return code: 161001" → 正确版有防护，未触发bug
A07: "Return status: 161001" → 正确版有防护，buggy版会返回0
A08: "SEGFAULT triggered due to null workspaceSize" → 触发了真实的空指针bug
```

## 结论

1. **逻辑矛盾类 bug 检出率 100%** (5/5)：Agent 通过对称性分析高效发现
2. **静态数据结构变更类检出率 0%** (0/2)：白名单增删需要硬件领域知识
3. **删除防护行类检出不稳定** (0/1)：A08 的 MAX_DIM 删除未被发现
4. **验证代码工程质量优秀**：100% 编译通过，NPU 上真实可运行
5. **意外收获**：A08 测试意外触发了 aclnnMulsGetWorkspaceSize 的真实空指针 bug
