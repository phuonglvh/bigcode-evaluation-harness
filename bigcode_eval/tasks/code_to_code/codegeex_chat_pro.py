from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import requests
import json
import sseclient
import os


def codegeex_chat_pro_translate(source_code, target_language="Java"):
    # Define the URL
    url = 'https://codegeex.cn/prod/code/chatCodeSseV3/chat'

    # Define the headers
    headers = {
        'accept': 'text/event-stream',
        'accept-language': 'en-US',
        'code-token': os.environ.get('CODEGEEX_CHAT_PRO_TOKEN'),
        'content-type': 'application/json',
    }

    # Define the body of the request
    body = {
        "lang": target_language,
        "machineId": "0b5607b5252bf2b518f0a541db6c2b8ef6f1699fdda8fe1c3ce378854cf2add3",
        "history": [],
        "command": "translation",
        "prompt": "",
        "locale": "en",
        "model": "codegeex-chat-pro",
        "code": source_code,
        "seed": 0
    }

    # Make the POST request
    response = requests.post(url, json=body, headers=headers, stream=True)

    # Check if the response is successful
    if response.status_code != 200:
        print(f"Request failed with status code: {response.status_code}")
        print("Response body:", response.text)
        return None

    # Create an Standard-Stream-Event client for processing the streaming response
    client = sseclient.SSEClient(response)

    translated_code = ""
    for event in client.events():
        try:
            event_data = json.loads(event.data)
            if event.event == "add":
                translated_code += event_data.get("text", "")
            elif event.event == "finish":
                # If you want to do something with the finished translation, you can add it here
                break
        except json.JSONDecodeError:
            print("Error decoding JSON:", event.data)

    return translated_code


def codegeex_chat_pro_translate_and_postprocess(translated_prompts_path, save_translations_path, limit_start=None, limit=None, parallel=False, max_workers=5):
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
        translated_code = codegeex_chat_pro_translate(prompt)
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
            translated_code = codegeex_chat_pro_translate(prompt)
            for text, new_text in replacements:
                translated_code = translated_code.replace(text, new_text)

            translated_code = java_imports + '\n' + translated_code
            translated_codes.append([translated_code])

        print(f'Translated {len(translated_codes)} prompts in serial')

    with open(save_translations_path, 'w') as f:
        f.write(json.dumps(translated_codes, indent=2))
        print(f'saved #{len(translated_codes)} translations to {save_translations_path}')
