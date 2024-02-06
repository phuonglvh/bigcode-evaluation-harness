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
        e.g. {bugfix-v2-multiple-py: Task, bugfix-v2-multiple-java: Task}
    """
    return {f"bugfix-v2-multiple-{language}": create_task(language) for language in LANGUAGE_ALIASES}


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
            "--debug",
            type=bool,
            help=f"debug mode to print more logs",
            required=False,
            default=False
        )
    return main_parser

def create_task(language):
    class MultiPLE(BugFixMultiPLE):
        def __init__(self, **kwargs):
            super().__init__(language, **kwargs)

    return MultiPLE


class BugFixMultiPLE(GeneralMultiPLE):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "nuprl/MultiPL-E"
    DATASET_NAME = None
    DATASET_REVISION = "d23b094346c5dbda1080a74bb2a24c18adbf7409"

    def __init__(self, language, **kwargs):
        assert language in LANGUAGE_ALIASES
        super().__init__(language)
        self.kwargs = kwargs
        self.bugfix_dataset: Dataset = None
        self.humaneval_dataset: Dataset = load_dataset(
            "openai_humaneval")['test']
        self.humaneval_dataset_dict = {}
        for task in self.humaneval_dataset:
            task_id = task['task_id'].replace('HumanEval/', '')
            self.humaneval_dataset_dict[task_id] = task

    def _get_prompt_bugfix(self, doc, source_code):
        entry_point_python = doc['entry_point']
        entry_point = entry_point_python
        if self.language.capitalize() == 'Java':
            entry_point = "".join([word.capitalize() for word in entry_point_python.split('_')])

        prompt = f'''Problem:
{source_code.rstrip()}\n    }}\n\n}}\n

Fix bugs in method {entry_point} if any:
{doc['prompt']}'''

        return prompt

    def _get_task_entrypoint(self, task_id: int) -> str:
        entry_point = self.humaneval_dataset_dict[task_id]['entry_point']
        return entry_point

    def get_dataset(self):
        """Returns the translated dataset for the task or an iterable of any object, that get_prompt can handle"""
        if self.bugfix_dataset:
            return self.bugfix_dataset

        source_generations_path = self.kwargs.get(
            "source_generations_path", None)

        tasks_generations = json.load(
            open(source_generations_path, 'r'))  # [["generated code"]]

        bugfix_tasks = []

        for task_gens, doc in zip(tasks_generations, self.dataset["test"]):
            for gen_id, task_gen in enumerate(task_gens):
                task_id = doc['name'].split('_')[1]
                entrypoint = self._get_task_entrypoint(task_id)

                sub_target_task = doc.copy()
                sub_target_task['entry_point'] = entrypoint
                sub_target_task['original_prompt'] = sub_target_task["prompt"]
                sub_target_task['prompt'] = self._get_prompt_bugfix(sub_target_task, task_gen)

                sub_target_task['original_name'] = sub_target_task["name"]
                sub_target_task['name'] = f'{sub_target_task["original_name"]}_{gen_id}'

                bugfix_tasks.append(sub_target_task)

        self.bugfix_dataset = Dataset.from_list(bugfix_tasks)
        return self.bugfix_dataset

    def get_prompt(self, doc) -> List[str]:
        """Builds the prompt for the LM to generate from."""
        prompt =  doc["prompt"].strip()

        if self.kwargs.get("debug", False):
            print(f'get_prompt:\n{prompt}')

        return prompt

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
        result = target_lang_prompt + self._stop_at_stop_token(completion, self.stop_words)
        if self.kwargs.get('debug', False):
            print(f'postprocess_generation:\ncompletion={completion}\nprocessed={result}')

        return result
