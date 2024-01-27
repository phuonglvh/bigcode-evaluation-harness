import argparse
import json
import re
from datasets import load_dataset


def main():
    lang_choices = ['java', 'python']
    parser = argparse.ArgumentParser(
        description='Convert humaneval-x to BLEU score references. Only humaneval-x items belong to generations are converted.')
    parser.add_argument('--language', '-lang',
                        required=True, help=f'The language of the generations. One of {lang_choices}.', choices=lang_choices)
    parser.add_argument('--load_generations_path', required=True,
                        help='absolute path to the generations json file')
    parser.add_argument('--save_references_path', required=True,
                        help='absolute path where you want to save their references in **jsonl** format')
    args = parser.parse_args()

    if args.language == 'python':
        dataset = load_dataset('openai_humaneval')['test']
    elif args.language == 'java':
        dataset = load_dataset('THUDM/humaneval-x', args.language)['test']
    else:
        raise ValueError('Unsupported language: {}'.format(args.language))

    all_gens = json.load(open(args.load_generations_path, 'r'))
    print(
        f'loaded generations {len(all_gens)} tasks from {args.load_generations_path}')

    if len(all_gens) > len(dataset):
        print(f'{len(all_gens)} unknown tasks will be NOT be converted')

    print(f'generations of {len(all_gens)} tasks will be converted')

    json_references = []
    for _, ds_task in zip(all_gens, dataset):
        prompt = ds_task.get('prompt')
        canonical_solution = ds_task.get('canonical_solution')

        ref_json = {}
        # prompt = re.sub(r'\s+', ' ', prompt) # replace sequences of whitespaces with a single whitespace
        prompt = re.sub(r'\r|\n', ' ', prompt)  # remove newline characters
        prompt.strip()

        # canonical_solution = re.sub(r'\s+', ' ', canonical_solution) # replace sequences of whitespaces with a single whitespace
        # remove newline characters
        canonical_solution = re.sub(r'\r|\n', ' ', canonical_solution)
        canonical_solution.strip()

        ref_json['nl'] = prompt
        sep = ' '
        ref_json['code'] = f'{prompt}{sep}{canonical_solution}'

        json_references.append(ref_json)

    with open(args.save_references_path, 'w') as outfile:
        for ref_json in json_references:
            outfile.write(json.dumps(ref_json) + '\n')
        print(
            f'saved {len(txt_references)} BLEU references at {args.save_references_path}')


if __name__ == '__main__':
    main()
