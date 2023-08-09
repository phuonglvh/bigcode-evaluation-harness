# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py

# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.
TODO: Write a Short Description of the task.
Homepage: TODO: Add the URL to the task's Homepage here.
"""
from collections import defaultdict
from lm_eval.base import Task

from evaluate import load

from datasets import load_dataset
import numpy as np

# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""

TRANSFORMATION_CATEGORIES = [
    "format",
    "func_name",
    "natgen",
    "nlaugmenter"
]
NUM_SEEDS = 5


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {
        f"perturbed-humaneval-{category}": create_task(category)
        for category in TRANSFORMATION_CATEGORIES
    }


def create_task(category):
    class PerturbedHumanEval(GeneralPerturbedHumanEval):
        DATASET_NAME = category
        def __init__(self):
            super().__init__(category)
    return PerturbedHumanEval


class GeneralPerturbedHumanEval(Task):
    DATASET_PATH = "RaymondLi/perturbed_humaneval"
    

    def __init__(self, category):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"],
            requires_execution=True,
        )
        # Transformation category
        self.category = category
        self.filtered_dataset = self.dataset['test'].filter(lambda x: x["seed"] < NUM_SEEDS)

    def get_dataset(self):
        """
        Returns dataset for the task or an iterable of any object, that get_prompt can handle
        Only keep the first NUM_SEEDS seeds
        """
        return self.filtered_dataset

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["prompt"].strip()

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        Will be passed to the `process_results` function, and potentially saved.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: dict
        """
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        test_code = "\n" + test_func + "\n" + entry_point
        return {
            "task_id": doc["task_id"],
            "seed": doc["seed"],
            "perturbation_name": doc["perturbation_name"],
            "test_code": test_code
        }

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]
    
    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        prompt = self.get_prompt(self.filtered_dataset[idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        # TODO: define how the evaluation score is computed from list of \
        # generations and reference solutions
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(dict)
            list of dict containing refrences
        :return: dict[str: float]
        """
        # # Load from json TODO: remove this
        # with open("code_eval_results.json", "r") as f:
        #     import json
        #     detailed_results = json.load(f)
        code_metric = load("code_eval")
        results, detailed_results = code_metric.compute(
            references=[ref["test_code"] for ref in references],
            predictions=generations,
        )
        # Dump as json TODO: remove this
        # with open("code_eval_results.json", "w") as f:
        #     import json
        #     json.dump(detailed_results, f)

        # Compute robust-pass-at-1. For each transformation and each prompt, we have s=5 randomly perturbed prompts.
        # With a single sample per prompt, RP@1 on a given transformation is the fraction of examples where completions
        # for all the perturbed prompts are correct.
        # With n samples per prompt, https://arxiv.org/abs/2212.10264 defines RP@1 as the average of the 
        # 1/n * sum_{i=1}^n I(all s correct for generation-seed i) over all prompts.
        # An alternate could be the average of the
        # prod_{j=1}^s 1/n * sum_{i=1}^n I(j-th prompt correct for generation-seed i) over all prompts.

        # We compute RP@1 for each transformation
        # transformation -> problem -> seed -> [n results]
        transformation_problem_results = defaultdict(lambda: defaultdict(dict))
        for i, ref in enumerate(references):
            result = detailed_results[str(i)]
            result = [x[1]["passed"] for x in result]
            assert ref["seed"] not in transformation_problem_results[ref["perturbation_name"]][ref["task_id"]]
            transformation_problem_results[ref["perturbation_name"]][ref["task_id"]][ref["seed"]] = result

        rp1 = {}
        for transformation, problem_results in transformation_problem_results.items():
            res = {}
            res["robust-pass-at-1"] = sum(
                # results = {seed -> [n results]}
                # 1/n * sum_{i=1}^n I(all s correct for generation-seed i)
                float(all(results_)) / len(list(results.values())[0])
                for results in problem_results.values()
                for results_ in zip(*results.values())
            ) / len(problem_results)

            res["alt-robust-pass-at-1"] = sum(
                # results = {seed -> [n results]}
                # prod_{j=1}^s 1/n * sum_{i=1}^n I(j-th prompt correct for generation-seed i)
                np.prod([
                    np.mean(results[j])
                    for j in results
                ])
                for results in problem_results.values()
            ) / len(problem_results)
            rp1[transformation] = res

        return rp1
