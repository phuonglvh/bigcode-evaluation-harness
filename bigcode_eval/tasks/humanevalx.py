"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""

import json
import os
import re
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from time import time
from typing import List
import warnings

import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.multiple_metrics.evaluation import \
    evaluate_problem
from bigcode_eval.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import \
    for_file
from bigcode_eval.tasks.code_to_code import java, py
import warnings

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {humanevalx-python: Task, humanevalx-java: Task}
    """
    LANGUAGES = ['python', 'java']
    return {f"humanevalx-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class HumanEvalX(GeneralHumanEvalX):
        def __init__(self, **kwargs):
            super().__init__(language, **kwargs)

    return HumanEvalX


class GeneralHumanEvalX(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "THUDM/humaneval-x"

    def __init__(self, language, **kwargs):
        self.args = kwargs
        self.k = kwargs.get('k', [1, 10, 100])
        self.num_workers = kwargs.get('num_workers', 16)
        self.timeout = kwargs.get('timeout', 3.0)
        
        self.language_extractor_map = {
            'python': py.extract_function_name_from_prompt,
            'java': java.extract_function_name_from_prompt,
        }
        
        self.language = language
        self.DATASET_SUBSET = language
        self.DATASET_REVISION = "main"
        
        # we need the dataset to get stop words for each language
        self.dataset: Dataset = load_dataset(
            GeneralHumanEvalX.DATASET_PATH,
            self.DATASET_SUBSET,
            revision=self.DATASET_REVISION,
            trust_remote_code=kwargs.get('trust_remote_code')
        )
        
        self.dataset['test'].add_column('name', [problem['task_id'] for problem in self.dataset['test']])
        
        json_problems = json.load(open(
            '/Users/phuonglvh/projects/2170558-thesis-automatic-code-generation-using-machine-learning/bigcode-evaluation-harness/benchmark/datasets/humaneval-x/humanevalx-java-refined.json', 'r'))
        
        
        print('adding name to test dataset by using task_id')
        for prob in json_problems:
            prob['name'] = f"{prob['task_id'].replace('/', '_')}_{self.language_extractor_map[self.language](prob['prompt'])}"
            
        self.dataset['test'] = Dataset.from_list(json_problems)

        for problem in self.dataset['test']:
            print(f'prob name = {problem['name']}')

        self.stop_words = [
            "\n    }\n}"
        ]
        self.requires_execution = True
        # {function_name -> task}
        self.problem_map = self.build_problem_map(self.dataset['test'])
        
    def _formalize_problem_name(self, problem_name):
        return problem_name.lower().replace('_', '')
        
        
    def identify_doc(self, generation):
        print('identifying problem from generation')
        problem_name = self.language_extractor_map[self.language](generation)
        
        problem_name = self._formalize_problem_name(problem_name)
        print(f'identified problem: {problem_name}')
        return problem_name
    
    def get_doc(self, problem_name):
        print(f'getting problem from problem name: {problem_name}')
        if problem_name in self.problem_map:
            return self.problem_map[problem_name]
        else:
            warnings.warn(f'problem_name={problem_name} not found in: {self.problem_map.keys()}')
            return None
        
    def build_problem_map(self, test_dataset):
        # build prob_prompt={problem_name -> problem}
        problem_map = {}
        
        for problem in test_dataset:
            prob_prompt = problem['prompt']
            try:
                problem_name = self._formalize_problem_name(
                    self.language_extractor_map[self.language](prob_prompt))

                if problem_name in problem_map:
                    print(f'existing:{problem_map[problem_name]}')
                    print(f'current:{prob_prompt}')
                    raise Exception(f'problem_name={problem_name} already exists in: {problem_map.keys()}')

                problem_map[problem_name] = problem
            except Exception as ex:
                print(ex)
                print(prob_prompt)
                raise ex
                
        return problem_map

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.strip_prompt:
            return doc["prompt"].strip()
        else:
            return doc["prompt"]

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return '\n    }' + doc['test'].replace('public class Main {', '')


    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.get_doc(self.identify_doc(generation)))
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
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
        
        def adapt_classname_to_eval_x_script(code):
            """
            Adapt the class name to the eval_x script
            For example: bigcode_eval/tasks/custom_metrics/multiple_metrics/eval_java.py
            """
            return code.replace('Solution', 'Problem')

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
                "completions": [adapt_classname_to_eval_x_script(prompt_completion) for prompt_completion in prompt_completions],
                "tests": adapt_classname_to_eval_x_script(reference),
            }
            
            # fist_completion = prompt_completions[0] if len(prompt_completions) > 0 else None
            # if fist_completion and prompt not in fist_completion:
            #     warnings.warn(f"prompt mismatches generations[0]:\nprompt={prompt}\n\ngeneration={fist_completion}")
                
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
        max_workers = max(self.num_workers, cpu_count() - 1 if cpu_count() > 1 else 1)
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
            for k, v in zip(self.k, result)
            if k <= len(generations[0])
        }

        print('computed pass@k')
        return results
