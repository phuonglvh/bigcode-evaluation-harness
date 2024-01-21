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

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.multiple_metrics.evaluation import \
    evaluate_problem
from bigcode_eval.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import \
    for_file
from bigcode_eval.tasks.multiple import GeneralMultiPLE

_CITATION = """
@article{cassano2022scalable,
  title={A Scalable and Extensible Approach to Benchmarking NL2Code for 18 Programming Languages},
  author={Cassano, Federico and Gouwar, John and Nguyen, Daniel and Nguyen, Sydney and Phipps-Costin, Luna and Pinckney, Donald and Yee, Ming Ho and Zi, Yangtian and Anderson, Carolyn Jane and Feldman, Molly Q and others},
  journal={arXiv preprint arXiv:2208.08227},
  year={2022}
}
"""

LANGUAGES = [
    "py",
    "sh",
    "cpp",
    "cs",
    "d",
    "go",
    "java",
    "js",
    "jl",
    "lua",
    "pl",
    "php",
    "r",
    "rkt",
    "rb",
    "rs",
    "scala",
    "swift",
    "ts",
]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-enc-dec-py: Task, multiple-enc-dec-java: Task}
    """
    return {f"multiple-enc-dec-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class MultiPLE(GeneralMultiPLEEncDec):
        def __init__(self):
            super().__init__(language)

    return MultiPLE

def extract_text(prompt, remove_lines=True):
    token = '\"\"\"'
    start = token
    end = '>>>'

    start_idx = prompt.find(start) + len(start)
    end_idx = prompt.find(end)

    output = prompt[start_idx: end_idx]
    if remove_lines:
        output = output.replace('\n', ' ')
    output = re.sub(r"\s+", " ", output).strip()

    return output

class GeneralMultiPLEEncDec(GeneralMultiPLE):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    def __init__(self, language):
        super().__init__(language)
        self.stop_words.append("<|endoftext|>")
        print(f'stop_words={self.stop_words}')
        # self.stop_words.append(*["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert"])
        self.encoder_prompt_pattern = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:'
        self.decoder_prompt_pattern = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:{}'
    
    def get_prompt_base(self, doc):
        return doc["prompt"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt_base = self.get_prompt_base(doc)
        instruction = extract_text(prompt_base)
        prompt = self.decoder_prompt_pattern.format(instruction, prompt_base)
        print(f'get_prompt (decoder): {prompt}')
        return prompt

    def get_prompt_encoder(self, doc):
        """Encoder input for models with Enc-Dec architecture like CodeT5"""
        prompt_base = self.get_prompt_base(doc)
        instruction = extract_text(prompt_base)
        prompt = self.encoder_prompt_pattern.format(instruction)
        print(f'get_prompt_encoder: {prompt}')
        return prompt

    def remove_last_block(self, code, stop_words):
        """
        Adapted from https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L151
        """
        for w in stop_words:
            if w in code:
                code = code[:code.find(w)]

        ### Find the first occassion where a chain of { } is closed
        if self.language == "py":
            for i, line in enumerate(code.split("\n")):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return "\n".join(code.split("\n")[:i])
        elif self.language in ["java", "js", "go", "cpp", "rust"]:
            open_brackets = 2 if self.language == "java" else 1
            cut = False
            for i, c in enumerate(code):
                if c == '{':
                    open_brackets += 1
                elif c == '}':
                    open_brackets -= 1
                if open_brackets == 0:
                    code = code[:i+1]
                    cut = True
                    break
            if not cut:
                if self.language == "java":
                    main_pos = code.find("public static void main")
                    if main_pos != -1:
                        code = code[:main_pos] + '}'
                    if '}' in code:
                        code = code[:code.rfind('}')] + '}'
                    if code.count('{') - 1 == code.count('}'):
                        code += "\n}"
                elif '}' in code:
                    code = code[:code.rfind('}')] + '}'
        return code

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        gen = self.remove_last_block(generation[len(prompt):].rstrip(), self.stop_words)
        # Strip to maintain same behavior as with get_prompt
        doc_prompt = doc["prompt"].rstrip()
        sep = '' if doc_prompt.endswith('\n') else '\n'
        post_gen = doc_prompt + sep + \
            self._stop_at_stop_token(gen, self.stop_words)
        print(f'postprocess_generation:\n\ngeneration={generation}\n\npost_gen={post_gen}')
        return post_gen

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        # get prompts and problem names
        prompts_names = [
            {"prompt": doc["prompt"], "name": doc["name"]}
            for i, doc in enumerate(self.get_dataset())
            if i < len(generations)
        ]
        # a common temp dir for all the problems
        temp_dir = tempfile.gettempdir()
        list_files = []
        for (prompt_name, generation, reference) in zip(
            prompts_names, generations, references
        ):
            problem = {
                "name": prompt_name["name"],
                "language": self.language,
                "prompt": prompt_name["prompt"],
                "completions": generation,
                "tests": reference,
            }
            # each problem is save in a json file
            temp_file_name = os.path.join(temp_dir, f"{prompt_name['name']}.json")
            list_files.append(temp_file_name)
            with open(temp_file_name, "wt") as f:
                json.dump(problem, f)
        print(
            f"Saved {len(list_files)} problems in {temp_dir} for evaluation, each problem has {len(generations[0])} completions"
        )

        # execute the problems to evaluate them
        max_workers = cpu_count() - 1 if cpu_count() > 1 else 1
        for file in tqdm(list_files):
            evaluate_problem(temp_dir, file, max_workers)

        # compute pass@k scores
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
        return results
