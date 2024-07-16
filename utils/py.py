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

def py_detect_unknown_tasks(dataset: Dataset, generations: List[List[str]]) -> List[str]:
    unknown = []
    
    for gens_at_i, task_gens in enumerate(generations):
        first_gen = task_gens[0]
        func_name = extract_function_name_from_prompt(first_gen)

        if f'def {func_name}(' not in dataset[gens_at_i]['prompt']:
            unknown.append(
                (gens_at_i, build_py_func_with_empty_body(first_gen)))
            
    return unknown
