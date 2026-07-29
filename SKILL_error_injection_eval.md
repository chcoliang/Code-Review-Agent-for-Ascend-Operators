# Skill: 昇腾算子错误注入 → Agent 盲审 → NPU 实测验证

## 核心原则

**所有评测都必须在真实 NPU 上编译运行验证。不接受仅基于代码分析的判定。**

无论是 op_api、op_host 还是 op_kernel 层的 bug：
- Agent 必须给出可编译的验证代码
- 验证代码必须在 NPU 上实际运行
- 以 NPU 的实际输出作为判定依据

---

## 第一步：创建错误注入

在算子源码中注入一个 bug，将注入后的代码放入 `agent_arena/cases/` 对应目录。

```bash
mkdir -p agent_arena/cases/<layer>/<CASE_ID>/
# 复制原始代码
cp <baseline_code> agent_arena/cases/<layer>/<CASE_ID>/
# 应用注入修改
```

验证注入生效：`diff` 结果非空。

---

## 第二步：调用独立 Agent 盲审

使用 `spawn_agent` 创建全新无头 Agent，**只给代码 + prompt**。

**Prompt 模板（必须包含以下要素）：**

```
你是 NPU 算子代码审查专家。审查目标运行在 Ascend 910B, CANN 8.5.0。

审查流程：
1. 通读代码，理解结构和数据流
2. 逐函数检查：参数校验、同族对称性、类型推导、边界条件、错误路径
3. 列出所有发现的 bug

对每个 bug 要求：
1. 指出位置（文件:函数:行号）、类型、严重程度、描述
2. 给出能触发该 bug 的测试输入数据（具体的 shape、dtype、值）
3. **必须**写出完整可编译的 C++ 验证程序：
   - 程序在 NPU 上编译运行
   - 输出能证明 bug 存在（打印返回值/行为）
   - 注明"正确行为应该是什么"
   - 如果 bug 导致崩溃，程序应能触发 SEGFAULT

不需要给出修复方案。

环境信息：
| 项目 | 值 |
|------|-----|
| 硬件 | Ascend 910B |
| CANN | 8.5.0 |
| 头文件 | /usr/local/Ascend/cann-8.5.0/include |
| 库路径 | /usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 |
| 链接 | -lascendcl -lnnopbase -lopapi |
| errno 头 | aclnn/opdev/op_errno.h (含 ACLNN_SUCCESS=0 等) |
| 算子头文件示例 | aclnnop/aclnn_mul.h, aclnnop/aclnn_softmax.h, aclnnop/aclnn_gelu.h |

编译命令模板：
g++ -std=c++17 -o test test.cpp \
  -I/usr/local/Ascend/cann-8.5.0/include \
  -L/usr/local/Ascend/cann-8.5.0/lib64 \
  -L/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 \
  -lascendcl -lnnopbase -lopapi \
  -Wl,-rpath,/usr/local/Ascend/cann-8.5.0/lib64:/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 \
  -Wl,--allow-shlib-undefined

重要：这是独立审查任务，仅基于代码本身判断。

审查文件：<绝对路径>
输出写入：<绝对路径>/result.md
```

---

## 第三步：NPU 实测（强制要求）

### 3.1 提取验证代码

从 `result.md` 中提取每个 bug 的验证程序到 `npu_tests/<CASE>_bug1.cpp`。

### 3.2 修复常见编译问题

```bash
# 缺少 errno 定义
sed -i '1i #include "aclnn/opdev/op_errno.h"' <file>.cpp

# 头文件路径错误
sed -i 's|"aclnn/aclnn_mul.h"|"aclnnop/aclnn_mul.h"|g' <file>.cpp
sed -i 's|"aclnn/aclnn_softmax.h"|"aclnnop/aclnn_softmax.h"|g' <file>.cpp
sed -i 's|"aclnn/aclnn_gelu.h"|"aclnnop/aclnn_gelu.h"|g' <file>.cpp

# 类型名错误
sed -i 's/aclRet /aclError /g' <file>.cpp
```

### 3.3 编译（必须全部通过）

```bash
g++ -std=c++17 -o <CASE>_bug1 <CASE>_bug1.cpp \
  -I/usr/local/Ascend/cann-8.5.0/include \
  -L/usr/local/Ascend/cann-8.5.0/lib64 \
  -L/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 \
  -lascendcl -lnnopbase -lopapi \
  -Wl,-rpath,/usr/local/Ascend/cann-8.5.0/lib64:/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 \
  -Wl,--allow-shlib-undefined
```

**如果编译失败，必须修复后重试直到成功。编译不通过 = 评测无效。**

### 3.4 在 NPU 上运行（必须执行）

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/lib64:\
/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64:\
/usr/local/Ascend/driver/lib64:\
/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64/driver

timeout 60 ./<CASE>_bug1 2>&1
echo "exit=$?"
```

**每个 case 都必须有 NPU 运行的实际输出记录。没有 NPU 输出 = 评测无效。**

### 3.5 对于 op_host/op_kernel 层的 bug

即使 bug 在 tiling 或 kernel 层，也必须通过 op_api 层调用来间接验证：

```cpp
// 通过 aclnnXxxGetWorkspaceSize 调用触发 tiling 错误
// 通过 aclnnXxx 执行调用触发 kernel 错误
// 观察返回错误码或运行时异常
```

如果当前 NPU 上安装的是正确版算子（无法部署 buggy 版），则：
1. 运行测试记录正确版的行为（baseline）
2. 明确说明"在 buggy 版上，此测试的预期输出是 XXX"
3. 对比 baseline vs buggy 预期，判定测试方案是否有区分能力

---

## 第四步：判定结果

### 4.1 判定标准（必须有 NPU 输出支撑）

| 判定项 | 条件 | 证据要求 |
|------|------|------|
| Agent 检出 ✅ | result.md 中提到注入点函数/行号/现象 | 引用原文 |
| 验证通过 ✅ | NPU 输出与 Agent 预测一致 | 贴出实际 NPU 输出 |
| 验证失败 ❌ | NPU 输出无异常 | 贴出实际 NPU 输出 |
| 编译失败 ❌ | g++ 报错 | 贴出编译错误 |

### 4.2 不接受的判定方式

- ❌ "代码分析可以看出 bug" — 必须有运行输出
- ❌ "理论上 buggy 版会崩溃" — 必须实际跑过
- ❌ "编译成功即验证通过" — 必须运行并看输出

---

## 第五步：记录

每个 case 的记录必须包含：

```markdown
| Case | 注入错误 | Agent检出 | NPU编译 | NPU运行输出 | 能暴露bug | 说明 |
|------|---------|:---:|:---:|------|:---:|------|
| <ID> | <类型> | ✅/❌ | ✅/❌ | <实际输出摘要> | ✅/❌ | <分析> |
```

---

## 完整单次执行命令序列

```bash
CASE="S01"
CODE="$(pwd)/agent_arena/cases/nn_softmax/${CASE}/aclnn_softmax.cpp"
OUT="$(pwd)/agent_arena/output/nn_softmax/${CASE}/result.md"

# 1. 注入已完成

# 2. Agent 盲审
# spawn_agent(... 审查文件: $CODE, 输出: $OUT ...)

# 3. 提取验证代码
mkdir -p npu_tests
# 从 result.md 提取到 npu_tests/${CASE}_bug1.cpp

# 4. 修复编译问题
cd npu_tests
sed -i '1i #include "aclnn/opdev/op_errno.h"' ${CASE}_bug1.cpp

# 5. 编译（必须成功）
g++ -std=c++17 -o ${CASE}_bug1 ${CASE}_bug1.cpp \
  -I/usr/local/Ascend/cann-8.5.0/include \
  -L/usr/local/Ascend/cann-8.5.0/lib64 \
  -L/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 \
  -lascendcl -lnnopbase -lopapi \
  -Wl,-rpath,/usr/local/Ascend/cann-8.5.0/lib64:/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64 \
  -Wl,--allow-shlib-undefined

# 6. NPU 运行（必须执行）
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/lib64:...
timeout 60 ./${CASE}_bug1 2>&1
echo "exit=$?"

# 7. 记录 NPU 实际输出到 TRACKING
```

---

## 关键原则（强制）

1. **NPU 实测是硬性要求**：每个 case 都必须在 NPU 上编译运行，记录实际输出
2. **Agent 隔离**：全新 session，只看代码，不看注入信息
3. **不要求修复**：Agent 只需识别 bug + 给出验证方式（输入数据 + 验证代码）
4. **编译必须通过**：如果 Agent 代码编译失败，修复后重试
5. **超时保护**：NPU 测试 60 秒超时
6. **所有层都走 NPU**：即使是 op_host/op_kernel 层 bug，也通过 op_api 调用在 NPU 上触发
7. **记录真实输出**：TRACKING 和实验报告中必须包含 NPU 的实际打印输出
