import argparse
import json
import re

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Convert MultiPL-E to CodeBLEU score references')
    parser.add_argument('--language', '-lang',
                        required=True, help='')
    parser.add_argument('--num_of_problems', '-num_prob',
                        required=True, help='')
    parser.add_argument('--references_path', '-ref_path',
                        required=True, help='')
    args = parser.parse_args()

    if args.language == 'python':
        tasks = load_dataset('openai_humaneval')['test']
    elif args.language == 'java':
        tasks = load_dataset('THUDM/humaneval-x', args.language)['test']
    else:
        raise ValueError('Unsupported language: {}'.format(args.language))

    reference_lines = []
    for task in tasks:
        ref_json = {}
        prompt = task.get('prompt')
        canonical_solution = task.get('canonical_solution')

        # prompt = re.sub(r'\s+', ' ', prompt) # replace sequences of whitespaces with a single whitespace
        prompt = re.sub(r'\r|\n', ' ', prompt) # remove newline characters
        prompt.strip()

        # canonical_solution = re.sub(r'\s+', ' ', canonical_solution) # replace sequences of whitespaces with a single whitespace
        canonical_solution = re.sub(r'\r|\n', ' ', canonical_solution) # remove newline characters
        canonical_solution.strip()

        sep = ' '

        reference_lines.append(f'{prompt}{sep}{canonical_solution}')

    with open(args.references_path, 'w') as outfile:
        for line in reference_lines:
            outfile.write(line + '\n')


if __name__ == '__main__':
    main()
