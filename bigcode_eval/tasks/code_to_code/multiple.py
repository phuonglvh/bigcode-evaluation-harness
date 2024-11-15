"""MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation
https://arxiv.org/abs/2107.03374

MultiPL-E is a dataset for evaluating large language models for code generation that supports 18 programming languages.
It takes the OpenAI "HumanEval" and the MBPP Python benchmarks and uses little compilers to translate them to other languages.

Homepage: https://nuprl.github.io/MultiPL-E/
"""

import json
import os
import re
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from time import time
from typing import *

import numpy as np
from tqdm import tqdm

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.multiple_metrics.evaluation import \
    evaluate_problem
from bigcode_eval.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import \
    for_file
from bigcode_eval.tasks.multiple import GeneralMultiPLE
from datasets import Dataset, load_dataset
from .utils import remove_py_docstring, remove_java_comments_before_first_public_static_func
import warnings
import sys

# sys.path.append(os.path.abspath(os.path.join(
#     os.path.dirname(__file__), "../../../utils")))
from . import java, py

_CITATION = """
@article{cassano2022scalable,
  title={A Scalable and Extensible Approach to Benchmarking NL2Code for 18 Programming Languages},
  author={Cassano, Federico and Gouwar, John and Nguyen, Daniel and Nguyen, Sydney and Phipps-Costin, Luna and Pinckney, Donald and Yee, Ming Ho and Zi, Yangtian and Anderson, Carolyn Jane and Feldman, Molly Q and others},
  journal={arXiv preprint arXiv:2208.08227},
  year={2022}
}
"""

LANGUAGE_ALIASES = [
    "py",
    # "sh",
    # "cpp",
    # "cs",
    # "d",
    # "go",
    "java",
    # "js",
    # "jl",
    # "lua",
    # "pl",
    # "php",
    # "r",
    # "rkt",
    # "rb",
    # "rs",
    # "scala",
    # "swift",
    # "ts",
]

MODEL_LANGUAGE_NAMES = {
    # "cpp"   : "C++",
    # "go"    : "Go",
    "java": "Java",
    # "js"    : "JavaScript",
    "python": "Python",
    "py": "Python",
}


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {code2code-multiple-py: Task, code2code-multiple-java: Task}
    """
    return {f"code2code-multiple-{language}": create_task(language) for language in LANGUAGE_ALIASES}


def add_task_specific_args(main_parser):
    main_args, _ = main_parser.parse_known_args()
    all_task_names = list(create_all_tasks().keys())

    if main_args.tasks in all_task_names:
        main_parser.add_argument(
            "--source_generations_path",
            type=str,
            help="path of source language generations",
            required=True
        )

        main_parser.add_argument(
            "--source_lang",
            type=str,
            choices=LANGUAGE_ALIASES,
            help=f"source language alias, {LANGUAGE_ALIASES}",
            required=True
        )
    return main_parser

def create_task(language):
    class MultiPLE(Code2CodeMultiPLE):
        def __init__(self, **kwargs):
            super().__init__(language, **kwargs)

    return MultiPLE


class Code2CodeMultiPLE(GeneralMultiPLE):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "nuprl/MultiPL-E"
    DATASET_NAME = None
    DATASET_REVISION = "d23b094346c5dbda1080a74bb2a24c18adbf7409"

    def __init__(self, language, **kwargs):
        assert language in LANGUAGE_ALIASES
        super().__init__(language, **kwargs)
        self.kwargs = kwargs

        self.source_lang = self.kwargs.get("source_lang", None)
        assert self.source_lang
        
        self.dataset['test'] = Dataset.from_list(json.load(open(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'datasets', f"humaneval-{self.source_lang}-{language}-refined.json"), "r")))
        
        # {function_name -> task}
        self.problem_map = self.build_problem_map(self.dataset['test'])
        
        self.stop_words = self.dataset["test"][0]["stop_tokens"] + ["<file_sep>"]
        self.translated_dataset: Dataset = None
        
    def build_problem_map(self, test_dataset):
        # build target {function_name -> task}
        tgt_map = {}
        
        for tgt_task in test_dataset:
            tgt_prompt = tgt_task['prompt']
            try:
                java_func_name = java.extract_function_name_from_prompt(
                    tgt_prompt).lower()

                if java_func_name in tgt_map:
                    print(f'existing:{tgt_map[java_func_name]}')
                    print(f'current:{tgt_prompt}')
                    raise Exception(f'java_func_name={java_func_name} already exists')

                tgt_map[java_func_name] = tgt_task
            except Exception as ex:
                print(ex)
                print(tgt_prompt)
                
        return tgt_map

    def _get_prompt_translation(self, target_doc, source_code, source_lang_alias, target_lang_alias):
        assert source_lang_alias in LANGUAGE_ALIASES
        assert target_lang_alias in LANGUAGE_ALIASES
        prompt = f'''code translation
{MODEL_LANGUAGE_NAMES[source_lang_alias].capitalize()}:
{remove_py_docstring(source_code.rstrip())}
{MODEL_LANGUAGE_NAMES[target_lang_alias].capitalize()}:
{remove_java_comments_before_first_public_static_func(target_doc['prompt'])}'''
        return prompt
    
    def audit_translated_prompt(self, source_code, target_prompt):
        py_func_name = py.extract_function_name_from_prompt(source_code).replace('_', '').lower()
        java_func_name = java.extract_function_name_from_prompt(target_prompt).lower()
        
        if py_func_name != java_func_name:
            raise ValueError(f"Formalized function names of source and target prompts do not match: {py_func_name} <> {java_func_name}")

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        print(f'Code2CodeMultiPLE.get_reference: {doc["name"]}')
        
        return doc["tests"].replace("\n    }\n\n}\n", "\nSystem.out.println(\"All test cases passed\");\n    }\n\n}\n")

    def get_dataset(self):
        """Returns the translated dataset for the task or an iterable of any object, that get_prompt can handle"""
        if self.translated_dataset:
            return self.translated_dataset

        target_lang_tasks = self.dataset['test']
        
        with open(f'MultiPL-E-{self.language}.json', 'w') as f:
            json.dump([problem for problem in target_lang_tasks], f)

        source_lang = self.kwargs.get("source_lang", None)
        target_lang = self.language
        source_generations_path = self.kwargs.get("source_generations_path", None)

        assert source_lang, source_generations_path

        tasks_generations = json.load(
            open(source_generations_path, 'r'))  # [["generated code"]]

        translated_tasks = []
        
        for task_gens in tasks_generations:
            for gen_id, task_gen in enumerate(task_gens):
                src_func_name = py.extract_function_name_from_prompt(task_gen).replace('_', '').lower()

                print(f'task func name={src_func_name}')
                target_lang_task = self.problem_map[src_func_name]
                
                if target_lang_task is None:
                    print(f'no target task for source func name={src_func_name}')
                    raise Exception(f'no target task for source func name={src_func_name}')
                
                sub_target_task = target_lang_task.copy()
                print(f"task name={sub_target_task['name']}")

                sub_target_task['original_prompt'] = sub_target_task["prompt"]
                sub_target_task['prompt'] = self._get_prompt_translation(
                    target_lang_task, task_gen, source_lang, target_lang)

                sub_target_task['original_name'] = sub_target_task["name"]
                sub_target_task['name'] = f'{sub_target_task["original_name"]}_{gen_id}'

                self.audit_translated_prompt(
                    task_gen, target_lang_task['prompt'])
                translated_tasks.append(sub_target_task)

        translated_dataset_path = 'translated-dataset.json'
        with open(translated_dataset_path, 'w') as f:
            json.dump(translated_tasks, f)
            print(f'num of target_lang_tasks:{len(self.problem_map.keys())}')
            print(f'num of flatten translated_tasks:{len(translated_tasks)}')
            print(f'saved translated dataset to {translated_dataset_path}')
            
        translated_prompts_path = 'translated-prompts.json'
        with open(translated_prompts_path, 'w') as f:
            json.dump([task['prompt'] for task in translated_tasks], f)
            print(f'saved translated prompts to {translated_prompts_path}')

        self.translated_dataset = Dataset.from_list(translated_tasks)
        return self.translated_dataset

    def get_prompt(self, doc) -> List[str]:
        """Builds the prompt for the LM to generate from."""
        return doc["prompt"].strip()

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        completion = generation[len(prompt) :]

        target_lang_prompt = self.get_dataset()[idx]['original_prompt'].strip()
        return target_lang_prompt + self._stop_at_stop_token(completion, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        print(f'executing Code2CodeMultiPLE.process_results')
        
        # get prompts and problem names
        n_tasks = len(generations)
        from_idx = self.args['limit_start']
        to_idx = from_idx+n_tasks
        selected_tasks = self.get_dataset().select(range(from_idx, to_idx)).to_list()
        print(f'process_results of {len(selected_tasks)} selected problems:\n' +
              '\n'.join(task['name'] for task in selected_tasks))

        prompts_names = [
            {"prompt": doc["prompt"], "name": doc["name"]}
            for doc in selected_tasks
        ]
        # a common temp dir for all the problems
        temp_dir = tempfile.gettempdir()
        list_files = []
        for (prompt_name, prompt_completions, reference) in zip(
            prompts_names, generations, references
        ):
            prompt = prompt_name['prompt']
            problem = {
                "name": prompt_name["name"],
                "language": self.language,
                "prompt": prompt_name["prompt"],
                "completions": prompt_completions,
                "tests": reference,
            }
            
            fist_completion = prompt_completions[0] if len(prompt_completions) > 0 else None
            if fist_completion and prompt not in fist_completion:
                warnings.warn(f"prompt mismatches generations[0]:\nprompt={prompt}\n\ngeneration={fist_completion}")
                
            # each problem is save in a json file
            temp_file_name = os.path.join(temp_dir, f"{prompt_name['name']}.json")
            list_files.append(temp_file_name)
            with open(temp_file_name, "wt") as f:
                json.dump(problem, f)
            print(f'saved 1 problem in {temp_file_name}')
        print(
            f"Saved {len(list_files)} problems in {temp_dir} for evaluation, each problem has {len(generations[0])} completions"
        )

        # execute the problems to evaluate them
        max_workers = cpu_count() - 1 if cpu_count() > 1 else 1
        for file in tqdm(list_files):
            evaluate_problem(temp_dir, file, max_workers)

        # compute pass@k scores
        print('computing pass@k')
        print(f'*.results.json stored in: {temp_dir}')
        result_array = np.array(
            [for_file(p) for p in Path(temp_dir).glob("*.results.json")]
        )
        result = result_array.mean(axis=0)
        name = (
            temp_dir.split("/")[-1]
            if temp_dir.split("/")[-1] != ""
            else temp_dir.split("/")[-2]
        )
        results = {
            f"pass@{k}": v
            for k, v in zip([1, 10, 100], result)
            if k <= len(generations[0])
        }

        print('computed pass@k')
        return results
