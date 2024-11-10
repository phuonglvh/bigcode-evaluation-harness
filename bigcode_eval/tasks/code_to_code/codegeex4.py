import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import requests
import json
import sseclient


def codegeex4_v1(prompt):
    # Import inside the function to avoid any top-level conflicts
    from zhipuai import ZhipuAI

    # Initialize ZhipuAI client
    client = ZhipuAI(api_key=os.environ.get('ZHIPUAI_API_KEY'))

    # Make a completion request
    response = client.chat.completions.create(
        model="codegeex-4",
        # Added prompt content for the chat
        messages=[{"role": "user", "content": prompt}],
        extra={
            "target": {
                "path": "quick_sort.py",
                "language": "Python",
                "code_prefix": "def quick_sort(arr):\n    ",
                "code_suffix": ""
            },
            "contexts": []
        },
        top_p=0.7,
        temperature=0.9,
        max_tokens=1024,
        stop=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|observation|>"]
    )

    # Return the message content of the response
    return response.choices[0].message["content"]


def codegeex4_translate(prompt):
    url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
    # Define the headers
    headers = {
        'accept-language': 'en-US',
        'content-type': 'application/json',
        'Authorization': f'Bearer {os.environ.get("ZHIPUAI_API_KEY", '3df252c70b96ea14a2f98bea30a305e3.ArFFZLsD7749V5vJ')}'
    }

    # Define the body of the request
    body = {
        "model": "codegeex-4",
        "locale": "en",
        "messages": [{"role": "user", "content": prompt}],
        "top_p": 0.7,
        "temperature": 0.9,
        "max_tokens": 1024,
    }

    # Make the POST request
    response = requests.post(url, json=body, headers=headers, stream=False)
    translated_code = response.json()["choices"][0]["message"]["content"]
    
    return translated_code

def codegeex4_translate_and_postprocess(translated_prompts_path, save_translations_path, limit_start=None, limit=None, parallel=False, max_workers=5):
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
        translated_code = codegeex4_translate(prompt)
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
            translated_code = codegeex4_translate(prompt)
            for text, new_text in replacements:
                translated_code = translated_code.replace(text, new_text)

            translated_code = java_imports + '\n' + translated_code
            translated_codes.append([translated_code])

        print(f'Translated {len(translated_codes)} prompts in serial')

    with open(save_translations_path, 'w') as f:
        f.write(json.dumps(translated_codes, indent=2))
        print(f'saved #{len(translated_codes)} translations to {save_translations_path}')


if __name__ == '__main__':
    print(codegeex4_translate("\ncode translation. Keep the target language imports and declarations.\nPython:\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n\nJava:\nimport java.util.*;\nimport java.lang.*;\n\nclass Solution {\n    public boolean hasCloseElements(List<Double> numbers, double threshold) {\n\n"))
