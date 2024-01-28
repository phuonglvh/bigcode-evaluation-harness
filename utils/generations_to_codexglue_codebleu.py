import argparse
import json
import re


def main():
    parser = argparse.ArgumentParser(
        description='Convert generations to CodeBLEU score predictions format')
    parser.add_argument('--load_generations_path', required=True, help="")
    parser.add_argument('--save_predictions_format_path',
                        required=True, help="")
    args = parser.parse_args()

    all_gens = json.load(open(args.load_generations_path, "r"))

    print(
        f'loaded generations of {len(all_gens)} tasks from {args.load_generations_path}')

    print(f'generations of {len(all_gens)} tasks will be converted')

    predictions = []
    for generations in all_gens:
        pred = generations[0].strip()  # use first sample
        # pred = re.sub(r'\s+', ' ', pred) # replace sequences of whitespaces with a single whitespace
        pred = re.sub(r'\r|\n', ' ', pred)  # remove newline characters
        predictions.append(pred)

    with open(args.save_predictions_format_path, 'w') as fp:
        for pred in predictions:
            fp.write(pred + '\n')

    print(
        f'saved {len(predictions)} CodeBLEU predictions at {args.save_predictions_format_path}')


if __name__ == '__main__':
    main()
