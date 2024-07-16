def extract_humaneval_task_id(task_name: str) -> int:
    return int(task_name.split('_')[1])
