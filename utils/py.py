import re
from typing import List
from datasets import load_dataset, Dataset

def build_py_func_with_empty_body(prompt: str) -> str:
    # def strlen(string: str) -> int: pass
    return prompt[(prompt.index('def ')):(prompt.index(':\n') + 1)] + ' pass'

def extract_function_name_from_prompt(prompt: str) -> str:
    func = build_py_func_with_empty_body(prompt)
    prefix = 'def '
    return func[func.index(prefix) + len(prefix):func.index('(')]

def py_detect_unknown_tasks(dataset: Dataset, generations: List[List[str]], start_idx: int) -> List[str]:
    assert dataset != None, 'Dataset must not be none'
    assert len(generations) > 0, 'generations must has at least one element'
    assert start_idx >= 0, 'start_idx must be greater than or equal to 0'

    unknown = []
    
    for gens_at_i, task_gens in enumerate(generations):
        first_gen = task_gens[0]
        func_name = extract_function_name_from_prompt(first_gen)
        
        dataset_task_idx = gens_at_i + start_idx

        if f'def {func_name}(' not in dataset[dataset_task_idx]['prompt']:
            unknown.append(
                (dataset_task_idx, build_py_func_with_empty_body(first_gen)))
            
    return unknown
