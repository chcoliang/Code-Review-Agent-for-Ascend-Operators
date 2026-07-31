#!/usr/bin/env python3
"""
Agent Arena Evaluation Framework
=================================
设计原则：
1. 每个 agent 只审查一个 case
2. agent 可读的范围：prompt + case 文件夹内的文件内容（内联在 prompt 中）
3. agent 不能访问外部文件系统路径

Usage:
    python3 scan_cases.py                    # 列出所有待扫描 case
    python3 scan_cases.py --generate         # 生成所有 case 的 prompt（JSON 输出）
    python3 scan_cases.py --case <op>/<id>   # 生成单个 case 的 prompt
"""
import os
import sys
import json

REPO = "/mnt/model/chencongliang/ccl/Code-Review-Agent-for-Ascend-Operators"
CASES_DIR = f"{REPO}/agent_arena/cases"
OUTPUT_DIR = f"{REPO}/agent_arena/output"

# ============================================================
# 评估 Prompt 模板
# 注意：agent 只能看到这个 prompt 和其中内联的代码
# 不暴露文件系统路径、不暴露其他 case 的信息
# ============================================================
REVIEW_PROMPT_TEMPLATE = """你是 Ascend NPU 算子代码审查专家。以下是一份运行在 Ascend 910B NPU 上的算子实现代码。

请系统审查代码，找出所有 bug。

## 审查方法

1. 通读代码，理解结构和数据流
2. 逐函数检查：
   - 参数校验是否完整（null、dtype、shape、range）
   - 类型推导和转换是否正确
   - 边界条件（空输入、极值、特殊类型）
   - 错误处理路径是否完备
   - 资源管理（内存分配/释放、流同步）
   - 精度和计算逻辑正确性
3. 列出所有发现的问题

## 待审查代码

{code_section}

## 输出要求

对每个发现的 bug，输出：

### Bug N: <简短描述>
- **位置**: <文件名>:<行号>
- **类型**: <bug 类别>
- **严重程度**: 高/中/低
- **描述**: <详细说明为什么这是 bug>
- **触发条件**: <构造什么输入可以触发>
- **测试方案**: <如何验证这个 bug>

最后输出汇总表：

| # | 位置 | 类型 | 严重程度 | 描述 |
|---|------|------|---------|------|
"""


def get_all_cases():
    """获取所有有代码文件的 case 目录"""
    cases = []
    for op_dir in sorted(os.listdir(CASES_DIR)):
        op_path = os.path.join(CASES_DIR, op_dir)
        if not os.path.isdir(op_path):
            continue
        for case_id in sorted(os.listdir(op_path)):
            case_path = os.path.join(op_path, case_id)
            if not os.path.isdir(case_path):
                continue
            code_files = sorted([f for f in os.listdir(case_path)
                                if f.endswith('.cpp') or f.endswith('.h')])
            if code_files:
                cases.append({
                    'op': op_dir,
                    'case_id': case_id,
                    'path': case_path,
                    'files': code_files,
                    'output_dir': os.path.join(OUTPUT_DIR, op_dir, case_id)
                })
    return cases


def generate_prompt_for_case(case):
    """
    为单个 case 生成 prompt。
    代码内联在 prompt 中，agent 不需要也不能读取外部文件。
    """
    code_sections = []
    for filename in case['files']:
        filepath = os.path.join(case['path'], filename)
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()
        code_sections.append(f"**文件: `{filename}`**\n\n```cpp\n{code}\n```")

    code_section = "\n\n".join(code_sections)
    return REVIEW_PROMPT_TEMPLATE.format(code_section=code_section)


def get_pending_cases():
    """获取尚未有 result.md 的 case"""
    cases = get_all_cases()
    return [c for c in cases
            if not os.path.exists(os.path.join(c['output_dir'], 'result.md'))]


def main():
    if '--case' in sys.argv:
        # 生成单个 case 的 prompt
        idx = sys.argv.index('--case')
        target = sys.argv[idx + 1]  # e.g., "nn_softmax/A19"
        op, cid = target.split('/')
        cases = get_all_cases()
        case = next((c for c in cases if c['op'] == op and c['case_id'] == cid), None)
        if not case:
            print(f"Case not found: {target}")
            sys.exit(1)
        prompt = generate_prompt_for_case(case)
        print(prompt)

    elif '--generate' in sys.argv:
        # 生成所有待处理 case 的 prompt（JSON 格式）
        pending = get_pending_cases()
        output = []
        for case in pending:
            prompt = generate_prompt_for_case(case)
            output.append({
                'case': f"{case['op']}/{case['case_id']}",
                'files': case['files'],
                'prompt': prompt,
                'output_file': os.path.join(case['output_dir'], 'result.md')
            })
        json.dump(output, sys.stdout, ensure_ascii=False, indent=2)

    else:
        # 显示状态
        all_cases = get_all_cases()
        pending = get_pending_cases()
        print(f"Total cases with code: {len(all_cases)}")
        print(f"Already reviewed: {len(all_cases) - len(pending)}")
        print(f"Pending: {len(pending)}")
        print()
        if pending:
            from collections import Counter
            ops = Counter(c['op'] for c in pending)
            print("Pending by operator:")
            for op, cnt in ops.most_common():
                print(f"  {op}: {cnt}")


if __name__ == '__main__':
    main()
