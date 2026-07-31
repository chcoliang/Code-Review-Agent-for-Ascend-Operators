#!/usr/bin/env python3
"""
Batch code review scanner for agent_arena cases.
Generates review prompts and outputs result.md for each case.
Usage: python3 scan_cases.py [start_idx] [count]
"""
import os
import sys
import json

REPO = "/mnt/model/chencongliang/ccl/Code-Review-Agent-for-Ascend-Operators"
CASES_DIR = f"{REPO}/agent_arena/cases"
OUTPUT_DIR = f"{REPO}/agent_arena/output"

REVIEW_PROMPT = """你是 NPU 算子代码审查专家。请审查以下 Ascend 910B 算子代码，找出所有 bug。

**任务**：
1. 找出代码中的所有 bug（参数校验、类型处理、边界条件、逻辑错误、资源管理、精度问题等）
2. 对每个 bug 给出：位置(行号)、类型、严重程度(高/中/低)、描述
3. 为每个 bug 设计一个触发测试方案（描述输入条件即可，不需要完整代码）

**重要**：仅基于代码本身分析，不要假设任何外部上下文。

**代码文件**: {filepath}

```cpp
{code}
```

请输出如下格式的 Markdown：

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
"""

def get_all_cases():
    """Get all case directories with code files."""
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
            if code_files:
                cases.append({
                    'op': op_dir,
                    'case_id': case_id,
                    'path': case_path,
                    'files': code_files,
                    'output_dir': os.path.join(OUTPUT_DIR, op_dir, case_id)
                })
    return cases

def generate_prompt(case):
    """Generate review prompt for a case."""
    all_code = ""
    for f in case['files']:
        filepath = os.path.join(case['path'], f)
        with open(filepath, 'r') as fh:
            code = fh.read()
        all_code += f"\n// === {f} ===\n{code}\n"
    
    filename = case['files'][0] if len(case['files']) == 1 else ', '.join(case['files'])
    filepath = os.path.join(case['path'], case['files'][0])
    
    return REVIEW_PROMPT.format(
        filepath=filepath,
        code=all_code.strip(),
        filename=filename
    )

def main():
    cases = get_all_cases()
    print(f"Total cases with code: {len(cases)}")
    
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else len(cases)
    
    batch = cases[start:start+count]
    
    # Output case list as JSON for batch processing
    output = []
    for case in batch:
        os.makedirs(case['output_dir'], exist_ok=True)
        prompt = generate_prompt(case)
        output.append({
            'op': case['op'],
            'case_id': case['case_id'],
            'output_file': os.path.join(case['output_dir'], 'result.md'),
            'prompt': prompt,
            'files': case['files']
        })
    
    # Write batch file
    batch_file = f"/tmp/review_batch_{start}_{start+count}.json"
    with open(batch_file, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Batch {start}-{start+count}: {len(output)} cases")
    print(f"Written to: {batch_file}")
    
    # Also print summary
    for item in output:
        print(f"  {item['op']}/{item['case_id']}: {', '.join(item['files'])}")

if __name__ == '__main__':
    main()
