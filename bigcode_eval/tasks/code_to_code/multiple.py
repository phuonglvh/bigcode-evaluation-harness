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
        super().__init__(language)
        self.kwargs = kwargs
        self.translated_dataset: Dataset = None

    def _get_prompt_translation(self, target_doc, source_code, source_lang_alias, target_lang_alias):
        assert source_lang_alias in LANGUAGE_ALIASES
        assert target_lang_alias in LANGUAGE_ALIASES
        prompt = f'''code translation
{MODEL_LANGUAGE_NAMES[source_lang_alias].capitalize()}:
{source_code.rstrip()}
{MODEL_LANGUAGE_NAMES[target_lang_alias].capitalize()}:
{target_doc['prompt']}'''
        return prompt

    def get_dataset(self):
        """Returns the translated dataset for the task or an iterable of any object, that get_prompt can handle"""
        if self.translated_dataset:
            return self.translated_dataset

        target_lang_tasks = self.dataset["test"]

        source_lang = self.kwargs.get("source_lang", None)
        target_lang = self.language
        source_generations_path = self.kwargs.get(
            "source_generations_path", None)

        assert source_lang, source_generations_path

        tasks_generations = json.load(
            open(source_generations_path, 'r'))  # [["generated code"]]

        translated_tasks = []

        for task_gens, target_lang_task in zip(tasks_generations, target_lang_tasks):
            for gen_id, task_gen in enumerate(task_gens):
                sub_target_task = target_lang_task.copy()
                sub_target_task['original_prompt'] = sub_target_task["prompt"]
                sub_target_task['prompt'] = self._get_prompt_translation(
                    target_lang_task, task_gen, source_lang, target_lang)

                sub_target_task['original_name'] = sub_target_task["name"]
                sub_target_task['name'] = f'{sub_target_task["original_name"]}_{gen_id}'

                translated_tasks.append(sub_target_task)

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
