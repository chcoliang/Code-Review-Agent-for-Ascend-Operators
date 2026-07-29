# Skill: 昇腾算子错误注入 → Agent 盲审 → NPU 验证

## 概述

完整流程：注入单个错误 → 调用独立 Agent 盲审识别 bug → Agent 给出验证数据和方法 → 在 NPU 上运行验证代码确认能检出 bug。

Agent 的目标是：**识别 bug + 给出能暴露 bug 的测试输入和预期异常输出**。不要求给出修复方案。

---

## 第一步：创建错误注入

在 `error_testset/op_api/<NN>_<error_id>/` 下准备：

```
README.md      — 错误描述（分类、位置、注入内容）
patch.diff     — 精确 diff
```

将注入后的代码放入盲审目录：

```bash
mkdir -p agent_arena/cases/op_api/A<NN>
cp error_testset/baseline/B01_no_inject/aclnn_mul.cpp agent_arena/cases/op_api/A<NN>/aclnn_mul.cpp
# 应用 patch
```

验证注入生效：`diff` 结果非空。

---

## 第二步：调用独立 Agent 盲审

使用 `spawn_agent` 创建全新无头 Agent，**只给代码文件 + 审查 prompt**，不给任何注入信息。

```python
spawn_agent(
  agent_type="default",
  message="""
你是 NPU 算子代码审查专家。审查目标运行在 Ascend 910B, CANN 8.5.0。

审查流程：
1. 通读代码，理解结构和数据流
2. 逐函数检查：参数校验、同族对称性、类型推导、边界条件、错误路径
3. 列出所有发现的 bug

对每个 bug 要求：
1. 指出位置（文件:函数:行号）、类型、严重程度、描述
2. 给出能触发该 bug 的测试输入数据（具体的 shape、dtype、值）
3. 写出完整可编译的 C++ 验证程序，程序输出能证明 bug 存在
   - 程序应打印实际返回值/行为
   - 并注明"正确行为应该是什么"
   - 如果 bug 会导致崩溃，程序应能触发 SEGFAULT

不需要给出修复方案。

环境信息：
| 项目 | 值 |
|------|-----|
| 硬件 | Ascend 910B |
| CANN | 8.5.0 |
| 头文件 | /usr/local/Ascend/cann-8.5.0/include |
| 库路径 | /usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 |
| 链接 | -lascendcl -lnnopbase -lopapi |
| errno 头 | aclnn/opdev/op_errno.h (含 ACLNN_SUCCESS 等定义) |
| mul 头 | aclnnop/aclnn_mul.h |

重要：这是独立审查任务，仅基于代码本身判断。

审查文件：<绝对路径>/agent_arena/cases/op_api/A<NN>/aclnn_mul.cpp
输出写入：<绝对路径>/agent_arena/output/op_api/A<NN>/result.md

输出格式：
### Bug N: <简短描述>
- **位置**: <文件:函数:行号>
- **类型**: <类别>
- **严重程度**: <高/中/低>
- **描述**: <说明>
- **触发输入**: <具体数据，如 shape=[2,3], dtype=FLOAT, out=nullptr>
- **预期异常**: <如 SEGFAULT / 返回错误码 / 计算结果错误>

#### 验证代码
```cpp
<完整可编译程序，输出能证明 bug>
```

最后输出汇总表（Bug编号、位置、触发条件、预期异常）。
""")
```

---

## 第三步：提取验证代码并在 NPU 上运行

### 3.1 提取

从 `result.md` 中提取每个 bug 的验证程序到 `npu_tests/A<NN>_bugN.cpp`。

### 3.2 修复编译问题（常见）

```bash
# 如果缺少 ACLNN_SUCCESS 等定义
sed -i '1i #include "aclnn/opdev/op_errno.h"' A<NN>_bugN.cpp

# 如果头文件路径错误
sed -i 's|"aclnn/aclnn_mul.h"|"aclnnop/aclnn_mul.h"|g' A<NN>_bugN.cpp
```

### 3.3 编译

```bash
g++ -std=c++17 -o A<NN>_bugN A<NN>_bugN.cpp \
  -I/usr/local/Ascend/cann-8.5.0/include \
  -L/usr/local/Ascend/cann-8.5.0/lib64 \
  -L/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 \
  -lascendcl -lnnopbase -lopapi \
  -Wl,-rpath,/usr/local/Ascend/cann-8.5.0/lib64:/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 \
  -Wl,--allow-shlib-undefined
```

### 3.4 运行

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/lib64:\
/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64:\
/usr/local/Ascend/driver/lib64:\
/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64/driver

timeout 60 ./A<NN>_bugN 2>&1
echo "exit=$?"
```

---

## 第四步：判定结果

### 4.1 Agent 是否识别出注入 bug

检查 `result.md` 中是否提到了注入点对应的**函数名**或**行号**或**等价现象描述**。

| 情况 | 判定 |
|------|------|
| 明确指出注入函数+问题 | ✅ 检出 |
| 描述了等价现象但未精确定位 | ⚠️ 部分检出 |
| 完全未提及 | ❌ 未检出 |

### 4.2 验证代码是否能暴露 bug

在 **buggy 版算子**上运行测试程序，观察输出是否符合 Agent 预测的异常行为：

| 运行结果 | 判定 |
|------|------|
| 输出与 Agent 预测一致（崩溃/错误码/计算错误） | ✅ 验证通过 |
| 程序正常运行但输出显示了错误行为 | ✅ 验证通过 |
| 程序正常运行且无异常 | ❌ 验证失败 |
| 编译失败 | ❌ 验证失败 |

### 4.3 对比 buggy vs 正确版（可选加强验证）

```bash
# 部署 buggy 版 → 运行 → 记录结果 B
bash deploy.sh <case_name>
./A<NN>_bugN > output_buggy.txt 2>&1

# 恢复正确版 → 运行 → 记录结果 C
bash deploy.sh restore
./A<NN>_bugN > output_correct.txt 2>&1

# 对比
diff output_buggy.txt output_correct.txt
# 有差异 = 测试有效
```

---

## 第五步：记录

更新 `TRACKING.md`：

```markdown
| case | 注入错误 | Agent检出 | 验证通过 | 说明 |
|------|---------|:---------:|:--------:|------|
| A<NN> | <类型> | ✅/❌ | ✅/❌ | <简述> |
```

---

## 完整单次执行命令序列

```bash
CASE="A09"
NN="09"
CODE_PATH="$(pwd)/agent_arena/cases/op_api/${CASE}/aclnn_mul.cpp"
OUT_PATH="$(pwd)/agent_arena/output/op_api/${CASE}/result.md"

# 1. 注入（已准备好 cases/op_api/A09/aclnn_mul.cpp）

# 2. 部署 buggy 版
bash deploy.sh 09_2.3_scalar_precision

# 3. 调用 Agent（spawn_agent，见第二步的 prompt）

# 4. 提取验证代码
mkdir -p npu_tests
# 从 result.md 提取 cpp 到 npu_tests/A09_bug1.cpp

# 5. 编译运行
cd npu_tests
bash build_all.sh
bash run_all.sh

# 6. 判定：输出是否显示了 bug（崩溃/错误返回/计算错误）
cat test_results.log
```

---

## 关键原则

1. **Agent 隔离**：全新 session，只看代码，不看注入信息
2. **不要求修复**：Agent 只需识别 bug + 给出验证方式（输入数据 + 预期异常）
3. **验证标准**：运行验证代码后输出能看出代码有问题即可
4. **超时保护**：NPU 测试 60 秒超时
5. **头文件提示**：prompt 中明确告知 Agent 正确的头文件路径和 errno 头，减少编译失败
