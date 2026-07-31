#!/usr/bin/env python3
"""
Batch code review runner using Kerminal sub-agents.
Generates task descriptions for processing cases in parallel batches.

Usage:
  python3 run_review_batch.py          # Show all pending cases
  python3 run_review_batch.py --batch N  # Generate batch N (3 cases each)
"""
import os
import sys
import json

REPO = "/mnt/model/chencongliang/ccl/Code-Review-Agent-for-Ascend-Operators"
CASES_DIR = f"{REPO}/agent_arena/cases"
OUTPUT_DIR = f"{REPO}/agent_arena/output"

PROMPT_TEMPLATE = """你是 NPU 算子代码审查专家。请审查以下 Ascend 910B 算子代码，找出所有 bug。

**任务**：
1. 找出代码中的所有 bug（参数校验、类型处理、边界条件、逻辑错误、资源管理、精度问题等）
2. 对每个 bug 给出：位置(行号)、类型、严重程度(高/中/低)、描述
3. 为每个 bug 设计一个触发测试方案（描述输入条件即可，不需要完整代码）

**重要**：仅基于代码本身分析，不要假设任何外部上下文。

请读取以下文件：{filepaths}

分析后将结果写入 `{output_file}`

输出格式：
```markdown
# Code Review: {filename}

## Bug 1: <简短描述>
- **位置**: 行号
- **类型**: <类别>
- **严重程度**: 高/中/低
- **描述**: <详细说明>
- **触发条件**: <如何构造输入触发此bug>
- **测试方案**: <测试思路>

## Bug N: ...

## 汇总
| # | 位置 | 类型 | 严重程度 | 描述 |
|---|------|------|---------|------|
```"""


def get_pending_cases():
    """Get cases that still need scanning."""
    cases = []
    for op_dir in sorted(os.listdir(CASES_DIR)):
        op_path = os.path.join(CASES_DIR, op_dir)
        if not os.path.isdir(op_path):
            continue
        for case_id in sorted(os.listdir(op_path)):
            case_path = os.path.join(op_path, case_id)
            if not os.path.isdir(case_path):
                continue
            code_files = [f for f in os.listdir(case_path)
                         if f.endswith('.cpp') or f.endswith('.h')]
            if not code_files:
                continue
            output_file = os.path.join(OUTPUT_DIR, op_dir, case_id, 'result.md')
            if os.path.exists(output_file):
                continue
            cases.append({
                'op': op_dir,
                'case_id': case_id,
                'path': case_path,
                'files': code_files,
                'output_file': output_file
            })
    return cases


def generate_agent_prompt(case):
    """Generate the prompt for a sub-agent."""
    filepaths = ', '.join(
        os.path.join(case['path'], f) for f in case['files']
    )
    filename = case['files'][0] if len(case['files']) == 1 else ', '.join(case['files'])
    return PROMPT_TEMPLATE.format(
        filepaths=filepaths,
        output_file=case['output_file'],
        filename=filename
    )


def main():
    cases = get_pending_cases()
    
    if '--batch' in sys.argv:
        batch_num = int(sys.argv[sys.argv.index('--batch') + 1])
        batch_size = 3
        start = batch_num * batch_size
        batch = cases[start:start + batch_size]
        
        print(f"Batch {batch_num}: cases {start}-{start+len(batch)-1}")
        print()
        for i, case in enumerate(batch):
            os.makedirs(os.path.dirname(case['output_file']), exist_ok=True)
            prompt = generate_agent_prompt(case)
            print(f"=== Agent {i+1}: {case['op']}/{case['case_id']} ===")
            print(f"Files: {', '.join(case['files'])}")
            print(f"Output: {case['output_file']}")
            print(f"Prompt length: {len(prompt)} chars")
            print()
    else:
        print(f"Pending cases: {len(cases)}")
        print(f"Batches needed (size 3): {(len(cases) + 2) // 3}")
        print()
        from collections import Counter
        ops = Counter(c['op'] for c in cases)
        for op, cnt in ops.most_common():
            print(f"  {op}: {cnt}")


if __name__ == '__main__':
    main()
