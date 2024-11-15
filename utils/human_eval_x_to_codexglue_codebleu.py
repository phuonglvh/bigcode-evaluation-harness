import argparse
import json
import re

from datasets import load_dataset


def main():
    lang_choices = ['java', 'python']
    parser = argparse.ArgumentParser(
        description='Convert humaneval-x to CodeBLEU score references. Only humaneval-x items belong to generations are converted.')
    parser.add_argument('--language', '-lang',
                        required=True, help=f'The language of the generations. One of {lang_choices}.', choices=lang_choices)
    parser.add_argument('--load_generations_path', required=True,
                        help='absolute path to the generations json file')
    parser.add_argument('--save_references_path', required=True,
                        help='absolute path where you want to save their references in **txt** format')
    args = parser.parse_args()

    language = args.language

    if language == 'python':
        # dataset = load_dataset('openai_humaneval')['test']
        dataset = load_dataset('THUDM/humaneval-x', name=language)['test']
    elif language == 'java':
        dataset = load_dataset('THUDM/humaneval-x', name=language)['test']
    else:
        raise ValueError('Unsupported language: {}'.format(language))

    multiple_e_ds = load_dataset(
        "nuprl/MultiPL-E", name=f"humaneval-{'py' if language == 'python' else 'java'}", revision="d23b094346c5dbda1080a74bb2a24c18adbf7409")['test']

    all_gens = json.load(open(args.load_generations_path, 'r'))
    print(
        f'loaded generations of {len(all_gens)} tasks from {args.load_generations_path}')

    assert len(all_gens) <= len(
        dataset), f'the number of tasks in generations ({len(all_gens)}) must be smaller dataset\'s ({len(dataset)})'

    txt_references = []
    for gen_i, task_gens in enumerate(all_gens):
        source_task = multiple_e_ds[gen_i] # order-wise
        task_id = source_task['name'].split('_')[1] # HumanEval_<task_id>_<func_name>
        humaneval_tasks = dataset.filter(lambda task: task['task_id'] == f'{language.capitalize()}/{task_id}') # task_id-wise

        assert len(humaneval_tasks) == 1, f'there must be only one humaneval_task, got {len(humaneval_tasks)} instead'
            
        ds_task = humaneval_tasks[0]
        assert ds_task, f'there must exist a dataset task for generation:\n{task_gens[0]}'

        prompt = ds_task.get('prompt')
        canonical_solution = ds_task.get('canonical_solution')

        # prompt = re.sub(r'\s+', ' ', prompt) # replace sequences of whitespaces with a single whitespace
        prompt = re.sub(r'\r|\n', ' ', prompt)  # remove newline characters
        prompt.strip()

        # canonical_solution = re.sub(r'\s+', ' ', canonical_solution) # replace sequences of whitespaces with a single whitespace
        # remove newline characters
        canonical_solution = re.sub(r'\r|\n', ' ', canonical_solution)
        canonical_solution.strip()

        sep = ' '

        txt_references.append(f'{prompt}{sep}{canonical_solution}')

    with open(args.save_references_path, 'w') as outfile:
        for line in txt_references:
            outfile.write(line + '\n')
        print(
            f'saved {len(txt_references)} CodeBLEU references at {args.save_references_path}')


if __name__ == '__main__':
    main()
