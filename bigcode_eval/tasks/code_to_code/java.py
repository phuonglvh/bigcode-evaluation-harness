import re
from typing import List
from datasets import load_dataset, Dataset

def build_java_public_static_func_with_empty_body(prompt: str) -> str:
    # return prompt[(prompt.index('public static ')):(prompt.index(') {') + 3)] + '}'
    # humaneval-x only
    prompt = prompt[prompt.index("class Solution"):]
    
    pattern_1 = ') {'
    pattern_2 = '){'
    pattern_3 = ') throws'
    ending_idx = -1

    if pattern_1 in prompt:
        ending_idx = prompt.rindex(pattern_1) + len(pattern_1)
        return prompt[(prompt.rindex('public ')):ending_idx] + '}'
    elif pattern_2 in prompt:
        ending_idx = prompt.rindex(pattern_2) + len(pattern_2)
        return prompt[(prompt.rindex('public ')):(ending_idx + len(pattern_2))] + ' }'
    elif pattern_3 in prompt:
        ending_idx = prompt.rindex(pattern_3)
        return prompt[(prompt.rindex('public ')):(ending_idx + len(pattern_3))] + ' { }'
    else:
        raise ValueError('Cannot find the end of the function body')

def extract_function_name_from_prompt(prompt: str) -> str:
    func = build_java_public_static_func_with_empty_body(prompt)
    print(f'prompt="{prompt}"')
    print(f'func="{func}"')
    prefixes = ['public static', 'public']
    start_idx = 0
    prefix = None
    for p in prefixes:
        if p in func:
            start_idx = func.index(p)
            prefix = p   
            break
        
    type_and_func_name = func[start_idx + len(prefix):func.index('(')]
    return type_and_func_name.split(' ')[-1]

def extract_function_name_from_prompt_v2(prompt: str) -> str:
    java_code = prompt
    pattern = r"""
        public\s+                # public keyword
        (?:static\s+)?           # optional static
        [\w<>\[\],\s]+?          # return type (có thể có generic, nhiều từ)
        (\w+)\s*                 # method name (capture group 1)
        \([^\)]*\)               # parameters (capture group 2)
        [\s\n]*                  # khoảng trắng hoặc xuống dòng
        \{?                      # dấu { nếu có (tùy chọn)
    """

    # Find all method signatures in class Solution, except main
    matches = list(re.finditer(pattern, java_code, re.VERBOSE))

    if len(matches) > 1:
        print(f'[WARNING] extract_function_name_from_prompt_v2: Found multiple function names in\n{prompt}')
        print(f'[WARNING] extract_function_name_from_prompt_v2: Matches={[m.group(0).strip() for m in matches]}')

    matched_names = []

    for match in matches:
        method_name = match.group(1)
        if method_name != "main":
            print(
                f'extract_function_name_from_prompt_v2: {match.group(0).strip()} => {method_name}')
        matched_names.append(method_name)
    
    if len(matched_names) == 0:
        print(f'[ERROR] extract_function_name_from_prompt_v2: Cannot find function name of\n{prompt}')
        return None
        
    return matched_names[-1]

def java_detect_unknown_tasks(dataset: Dataset, generations: List[List[str]], start_idx: int) -> List[str]:
    assert dataset != None, 'Dataset must not be none'
    assert len(generations) > 0, 'generations must has at least one element'
    assert start_idx >= 0, 'start_idx must be greater than or equal to 0'
    
    unknown = []

    for gens_at_i, task_gens in enumerate(generations):
        first_gen = task_gens[0]
        func_name = extract_function_name_from_prompt(first_gen)

        dataset_task_idx = gens_at_i + start_idx

        if f' {func_name}(' not in dataset[dataset_task_idx]['prompt']:
            unknown.append(
                (dataset_task_idx, build_java_public_static_func_with_empty_body(first_gen)))

    return unknown
