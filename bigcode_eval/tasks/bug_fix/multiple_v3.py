"""MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation
https://arxiv.org/abs/2107.03374

MultiPL-E is a dataset for evaluating large language models for code generation that supports 18 programming languages.
It takes the OpenAI "HumanEval" and the MBPP Python benchmarks and uses little compilers to translate them to other languages.

Homepage: https://nuprl.github.io/MultiPL-E/
"""

from typing import *

from bigcode_eval.tasks.bug_fix.multiple_v2 import \
    BugFixMultiPLE as BugFixV2MultiPLE
from bigcode_eval.tasks.bug_fix.utils import remove_comments

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
    "java",
]

MODEL_LANGUAGE_NAMES = {
    "java": "Java",
    "python": "Python",
    "py": "Python",
}

LANGUAGE_TAG = {
    "python": "# language: Python",
    "java": "// language: Java",
}


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {bugfix-v3-multiple-py: Task, bugfix-v3-multiple-java: Task}
    """
    return {f"bugfix-v3-multiple-{language}": create_task(language) for language in LANGUAGE_ALIASES}


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


class BugFixMultiPLE(BugFixV2MultiPLE):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    def __init__(self, language, **kwargs):
        assert language in LANGUAGE_ALIASES
        super().__init__(language, **kwargs)

    def get_prompt_bugfix(self, doc, source_code):
        entry_point_python = doc['entry_point']
        entry_point = entry_point_python

        words = entry_point_python.split('_')
        if self.language.capitalize() == 'Java' and len(words) > 1:
            entry_point = words[0] + "".join([word.capitalize() for word in words[1:]])

        prompt = f'''{source_code.rstrip()}\n    }}\n\n}}\n
Fix bugs in {entry_point}:
{remove_comments(doc["prompt"])}
'''

        return prompt
