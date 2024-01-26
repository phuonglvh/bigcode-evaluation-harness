import inspect
from pprint import pprint

from . import (apps, code_to_code, codexglue_code_to_text,
               codexglue_text_to_text, conala, concode, ds1000, gsm, humaneval,
               humanevalpack, instruct_humaneval, instruct_wizard_humaneval,
               mbpp, multiple, multiple_enc_dec, parity, python_bugs, quixbugs,
               recode, santacoder_fim)

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **codexglue_code_to_text.create_all_tasks(),
    **codexglue_text_to_text.create_all_tasks(),
    **multiple.create_all_tasks(),
    **multiple_enc_dec.create_all_tasks(),
    "codexglue_code_to_text-python-left": codexglue_code_to_text.LeftCodeToText,
    "conala": conala.Conala,
    "concode": concode.Concode,
    **ds1000.create_all_tasks(),
    **humaneval.create_all_tasks(),
    **humanevalpack.create_all_tasks(),
    "mbpp": mbpp.MBPP,
    "parity": parity.Parity,
    "python_bugs": python_bugs.PythonBugs,
    "quixbugs": quixbugs.QuixBugs,
    "instruct_wizard_humaneval": instruct_wizard_humaneval.HumanEvalWizardCoder,
    **gsm.create_all_tasks(),
    **instruct_humaneval.create_all_tasks(),
    **recode.create_all_tasks(),
    **santacoder_fim.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))

TRANSLATION_TASK_REGISTRY = {
    **code_to_code.multiple.create_all_tasks(),
}
TRANSLATION_TASKS = sorted(list(TRANSLATION_TASK_REGISTRY))


ALL_TASK_SPECIFIC_ARGS = [code_to_code.multiple.add_task_specific_args]

def get_task(task_name, args=None):
    if task_name in TRANSLATION_TASKS:
        return get_code_to_code_task(task_name, args)

    try:
        kwargs = {}
        if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["prompt"] = args.prompt
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["load_data_path"] = args.load_data_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_code_to_code_task(task_name, args=None):
    try:
        kwargs = {'source_lang': args.source_lang,
                  'source_generations_path': args.source_generations_path}
        if "prompt" in inspect.signature(TRANSLATION_TASK_REGISTRY[task_name]).parameters:
            kwargs["prompt"] = args.prompt
        if "load_data_path" in inspect.signature(TRANSLATION_TASK_REGISTRY[task_name]).parameters:
            kwargs["load_data_path"] = args.load_data_path
        return TRANSLATION_TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TRANSLATION_TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
