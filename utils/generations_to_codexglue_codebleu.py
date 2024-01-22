import argparse
import json
import re

def main():
    parser = argparse.ArgumentParser(
        description='Convert generations to CodeBLEU score predictions')
    parser.add_argument('--generations_path', '-gen_path',
                        required=True, help="")
    parser.add_argument('--predictions_path', '-pred_path',
                        required=True, help="")
    args = parser.parse_args()

    with open(args.generations_path, "r") as fp:
        generations_of_problems = json.load(fp)

    predictions = []
    for generations in generations_of_problems:
        pred = generations[0].strip()  # use first sample
        # pred = re.sub(r'\s+', ' ', pred) # replace sequences of whitespaces with a single whitespace
        pred = re.sub(r'\r|\n', ' ', pred) # remove newline characters
        predictions.append(pred)

    with open(args.predictions_path, 'w', encoding='utf-8') as fp:
        for pred in predictions:
            fp.write(pred + '\n')


if __name__ == '__main__':
    main()
