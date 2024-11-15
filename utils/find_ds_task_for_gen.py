from extract_func_info import extract_func_info

tabs = '    '


def build_dummy_func(func_decl, dummy_body=f'\n{tabs}pass'):
    return f'{func_decl}{dummy_body}'


end_of_func_decl = {
    'python': ':\n',
    'java': ') {\n'
}

dummy_gen = {
    'python': f'{tabs}pass',
    'java': f"{tabs}" + '}' + f"\n" + '}'
}


def find_ds_task_of_gen(dataset, task_gen, language):
    idx = task_gen.index(end_of_func_decl[language])
    task_gen = build_dummy_func(
        f'{task_gen[:idx]}{end_of_func_decl[language]}', dummy_body=dummy_gen[language])
    try:
        gen_func_info = extract_func_info(task_gen, language)
    except Exception as e:
        print(e)
        raise e

    gen_func_name = gen_func_info['function_name']
    gen_modifiers = gen_func_info['modifiers']
    gen_func_args = gen_func_info['arguments']
    gen_return_type = gen_func_info['return_type']

    if language == 'java':
        if 'static' in gen_modifiers:
            gen_modifiers = [
                modifier for modifier in gen_modifiers if modifier != 'static']
            gen_modifiers.append('static')

    func_decl = {
        'python': f'def {gen_func_name}(',
        'java': f'{" ".join(gen_modifiers)} {gen_return_type} {gen_func_name}('
    }

    ds_tasks = dataset.filter(lambda task: (
        func_decl[language] in task['prompt']))

    if len(ds_tasks) == 0:
        print(f'no dataset tasks matches {gen_func_info}')
        return None

    if len(ds_tasks) == 1:
        ds_task = ds_tasks[0]
    else:
        task_func_infos = [extract_py_func_info(
            build_dummy_func(ds_task["prompt"])) for ds_task in ds_tasks]

        for task_i, gen_func_info in enumerate(task_func_infos):
            if len(gen_func_info['arguments']) == len(gen_func_args):
                ds_task = ds_tasks[task_i]
                break

    return ds_task
