import inspect
from pprint import pprint

from . import code_to_code

TASK_REGISTRY = {
    **code_to_code.multiple.create_all_tasks(),
}

TRANSLATION_TASKS = TASK_REGISTRY

ALL_TASKS = sorted(list(TRANSLATION_TASKS))


def get_task(task_name, args=None):
    try:
        kwargs = {'source_lang': args.source_lang,
                  'source_generations_path': args.source_generations_path}
        if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["prompt"] = args.prompt
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["load_data_path"] = args.load_data_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
