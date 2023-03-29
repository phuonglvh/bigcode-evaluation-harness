"""WIP

Homepage: https://github.com/bigcode-project/commits
"""

import re
from evaluate import load
from lm_eval.base import Task


_CITATION = """
"""

class HumanEvalXBugs(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "Muennighoff/humaneval-x-bugs"
    DATASET_NAME = "python"

    def __init__(self, mutate_method="prompt"):
        
        stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
        self.mutate_method = mutate_method
        if self.mutate_method == "edit":
            stop_words = [
                "<commit_before>", 
                "<commit_msg>", 
                "<commit_after>", 
                "<|endoftext|>",
            ]

        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.mutate_method == "edit":
            prompt = "<commit_before>" + doc["buggy_solution"]
            prompt += "<commit_msg>" + "Fix bug in " + doc["entry_point"] # TODO Needs to be camel case if Java
            prompt += "<commit_after>"
        elif self.mutate_method == "edit-type":
            prompt = "<commit_before>" + doc["buggy_solution"]
            prompt += "<commit_msg>" + "Fix " + doc["bug_type"] + " in " + doc["entry_point"]
            prompt += "<commit_after>"
        elif self.mutate_method == "prompt":
            prompt = "# Buggy function"
            prompt += "\n" + doc["buggy_solution"] + "\n"
            prompt += "# Fixed function\ndef"            
        else:
            raise ValueError(f"Unknown mutate_method: {mutate_method}")

        return prompt.strip()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    @staticmethod
    def remove_last_block(string, stop_words):
        stop_words = [re.escape(word) for word in stop_words] # Escape e.g. | in <|endoftext|>
        # Remove the last block of the code containing stop_words for HumanEval
        string_list = re.split("(%s)" % "|".join(stop_words), string)
        # last string should be ""
        return "".join(string_list[:-2])

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
        generation = generation[len(prompt):]
        if self.mutate_method == "prompt":
            generation = "def" + generation # Add def which is in the prompt back to the output        
        return self.remove_last_block(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("code_eval")
        results, _ = code_metric.compute(
            references=references,
            predictions=generations,
        )
        return results
