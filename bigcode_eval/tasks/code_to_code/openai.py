from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import requests
import json
import sseclient
from bigcode_eval.tasks.openai import openai_chat_completions


def openai_translate(source_code, target_language="Java", model='gpt-4o'):
    return openai_chat_completions(
        [
            {'role': 'system', 'content': 'You are a superb code translator.'},
            {'role': 'user', 'content': source_code}
        ], 'gpt-4o')



def openai_translate_and_postprocess(translated_prompts_path, save_translations_path, limit_start=None, limit=None, parallel=False, max_workers=5, **kwargs):
    prompts = json.load(open(translated_prompts_path, "r"))
    
    translated_codes = []

    # Slice the prompts list based on the limit_start (optional), limit (optional)
    if limit_start is not None and limit is not None:
        selected_prompts = prompts[limit_start:limit_start + limit]
    else:
        selected_prompts = prompts

    # java_imports = 'import java.util.*;\nimport java.lang.reflect.*;\nimport org.javatuples.*;\nimport java.security.*;\nimport java.math.*;\nimport java.io.*;\nimport java.util.stream.*;'
    java_imports = ''

    replacements = [
        ('```', ''),
        ('java\n', ''),
        ("\n    }\n}\n", ''),
        ("public class", "class")
    ]

    # Worker function
    def process_prompt(prompt):
        translated_code = openai_translate(
            prompt, target_language="Java", model=kwargs.get('model', 'gpt-4o'))
        for text, new_text in replacements:
            translated_code = translated_code.replace(text, new_text)

        translated_code = java_imports + '\n' + translated_code
        return [translated_code]

    if parallel:
        print(f"Using parallel processing with #{max_workers} workers")
        # Using ThreadPoolExecutor to parallelize the loop
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map the worker function to selected_prompts and collect results
            translated_codes = list(executor.map(
                process_prompt, selected_prompts))

        print(f'Translated {len(translated_codes)} prompts in parallel')
    else:
        for prompt in tqdm(selected_prompts, desc="Translating Prompts", unit="prompt"):
            translated_code = openai_translate(prompt)
            for text, new_text in replacements:
                translated_code = translated_code.replace(text, new_text)

            translated_code = java_imports + '\n' + translated_code
            translated_codes.append([translated_code])

        print(f'Translated {len(translated_codes)} prompts in serial')

    with open(save_translations_path, 'w') as f:
        f.write(json.dumps(translated_codes, indent=2))
        print(f'saved #{len(translated_codes)} translations to {save_translations_path}')


if __name__ == "__main__":
    print(openai_chat_completions([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
    ]))
