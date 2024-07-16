from typing import List
from datasets import load_dataset, Dataset

def build_java_public_static_func_with_empty_body(prompt: str) -> str:
    return prompt[(prompt.index('public static ')):(prompt.index(') {') + 3)] + '}'

def extract_function_name_from_prompt(prompt: str) -> str:
    func = build_java_public_static_func_with_empty_body(prompt)
    prefix = 'public static'
    type_and_func_name = func[func.index(prefix) + len(prefix):func.index('(')]
    return type_and_func_name.split(' ')[-1]

def java_detect_unknown_tasks(dataset: Dataset, generations: List[List[str]]) -> List[str]:
    unknown = []

    for gens_at_i, task_gens in enumerate(generations):
        first_gen = task_gens[0]
        func_name = extract_function_name_from_prompt(first_gen)

        if f' {func_name}(' not in dataset[gens_at_i]['prompt']:
            unknown.append(
                (gens_at_i, build_java_public_static_func_with_empty_body(first_gen)))

    return unknown
