import argparse
import json
import re

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Convert humaneval-x to BLEU score references')
    parser.add_argument('--language', '-lang',
                        required=True, help='')
    parser.add_argument('--predictions_path', '-pred_path',
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

    func_sig_pattern = r'def\s+(\w+)\s*\('
    task_predictions = json.load(open(args.predictions_path, 'r'))
    task_predictions = [predictions[0] for predictions in task_predictions]

    predicted_funcs = []
    for task_prediction in task_predictions:
        match = re.search(func_sig_pattern, task_prediction)
        if match:
            function_name = match.group(1)
            predicted_funcs.append(function_name)
        else:
            raise ValueError(
                'Could not find function name in prediction: {}'.format(task_prediction))

    reference_jsons = []

    for predicted_func in predicted_funcs:
        task = [task for task in tasks if task['entry_point'] == predicted_func][0]
        prompt = task.get('prompt')
        canonical_solution = task.get('canonical_solution')

        ref_json = {}
        # prompt = re.sub(r'\s+', ' ', prompt) # replace sequences of whitespaces with a single whitespace
        prompt = re.sub(r'\r|\n', ' ', prompt) # remove newline characters
        prompt.strip()

        # canonical_solution = re.sub(r'\s+', ' ', canonical_solution) # replace sequences of whitespaces with a single whitespace
        canonical_solution = re.sub(r'\r|\n', ' ', canonical_solution) # remove newline characters
        canonical_solution.strip()


        ref_json['nl'] = prompt
        sep = ' '
        ref_json['code'] = f'{prompt}{sep}{canonical_solution}'

        reference_jsons.append(ref_json)

    with open(args.references_path, 'w') as outfile:
        for ref_json in reference_jsons:
            outfile.write(json.dumps(ref_json) + '\n')


if __name__ == '__main__':
    main()
