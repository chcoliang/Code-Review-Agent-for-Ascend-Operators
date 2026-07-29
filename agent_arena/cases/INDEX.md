# Case 索引

> **本文件仅人工维护，Agent 评测时不提供。**

## 映射表

| Agent ID | error_testset 名称 | 错误编号 | 错误类型 | 层 | 状态 |
|:--:|------|:--:|------|:--:|:--:|
| **baseline/B01** | B01_no_inject | — | 无注入（官方原始代码） | — | ✅ |
| **baseline/B02** | B02_no_inject | — | 无注入（独立重复实验） | — | ✅ |
| **op_api/A01** | 01_1.1_nullptr_check | 1.1 | 空指针校验缺失 | op_api | ✅ |
| **op_api/A02** | 02_4.3_shape_verify | 4.3 | 输出Shape校验缺失 | op_api | ✅ |
| **op_api/A03** | 03_2.1_mixdtype_removed | 2.1 | 混合精度组合被误删 | op_api | ⚠️ 分类存疑 |
| **op_api/A04** | 04_3.2_empty_tensor | 3.2 | 空Tensor处理遗漏 | op_api | ✅ |
| **op_api/A05** | 05_1.2_dtype_whitelist | 1.2 | dtype白名单遗漏 | op_api | ✅ |
| **op_api/A06** | 06_1.3_dtype_overwide | 1.3 | dtype白名单过宽 | op_api | ✅ |
| **op_api/A07** | 07_1.4_errorcode_masking | 1.4 | 错误码伪装 | op_api | ✅ |
| **op_api/A08** | 08_4.2_dim_check | 4.2 | 维度上限检查缺失 | op_api | ✅ |

## 目录结构

```
cases/
├── INDEX.md              ← 本文件
├── baseline/             ← 无注入对照
│   ├── B01/              ← 官方代码 run1
│   └── B02/              ← 独立重复 run2
├── op_api/               ← L2 API 层注入
│   ├── A01/ ~ A08/       ← 已完成
│   └── A09/ ...          ← 待注入
├── op_host/              ← Tiling + 注册层注入 (Phase 2)
└── op_kernel/            ← Kernel 计算层注入 (Phase 3)
```

## 输出

评测结果写入 `../output/`，目录名与 case ID 一一对应：

```
output/baseline/B01/result.md
output/op_api/A01/result.md
...
```

## 新增 Case 流程

1. 在 `error_testset/` 下按清晰命名创建（如 `op_api/1.4_errorcode_masking/`）
2. 自验证通过后，复制待审查代码到 `cases/op_api/A05/`
3. 更新本表，添加对应行
4. `bash deploy.sh A05` 部署到 NPU
5. Agent 审查 `cases/op_api/A05/`，输出到 `output/op_api/A05/`
