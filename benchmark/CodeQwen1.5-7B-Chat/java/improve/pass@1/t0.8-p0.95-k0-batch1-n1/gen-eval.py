# %%
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "info"

# %%
# imports
import json

import logging

# Set up logging configuration at the top of your notebook or script
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more verbosity
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/CodeQwen1.5-7B-Chat"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
import time

def generate_for_prompt(model, tokenizer, user_prompt, **kwargs):
    def build_messages(user_prompt):
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Respond naturally to the user's request. If the response contains code, output it raw without additional explanations."
                )
            },
            {"role": "user", "content": user_prompt}
        ]

    logging.info(kwargs)
    max_new_tokens = kwargs.get("max_new_tokens", 1024)
    top_p = kwargs.get("top_p", 0.95)
    temperature = kwargs.get("temperature", 0.1)
    top_k = kwargs.get("top_k", 0)

    num_return_sequences = kwargs.get("num_return_sequences", 1)
    do_sample = kwargs.get("do_sample", False)

    messages = build_messages(user_prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Measure execution time
    start_time = time.time()
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        top_p=top_p,
        temperature=temperature,
        top_k=top_k
    )

    execution_time = time.time() - start_time
    
    logging.info(f"⏱️ Took {execution_time:.2f} secs")

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    return response

# %%
from tqdm import tqdm
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_for_prompts_v2(model, tokenizer, prompts, chunk_size=4, max_workers=4, **kwargs):
    """
    Generate completions for prompts in parallel, divided into chunks.
    Reports overall progress across all prompts using tqdm.
    """
    total = len(prompts)
    completions = [None] * total
    num_chunks = math.ceil(total / chunk_size)
    logging.info(
        f"Total prompts: {total}, Chunk size: {chunk_size}, Chunks: {num_chunks}")

    def process_chunk(chunk_prompts, chunk_indices):
        chunk_results = []
        for idx, prompt in zip(chunk_indices, chunk_prompts):
            response = generate_for_prompt(model, tokenizer, prompt, **kwargs)
            chunk_results.append((idx, response))
        return chunk_results

    # Prepare chunks
    chunks = [
        (prompts[i:i+chunk_size], list(range(i, min(i+chunk_size, total))))
        for i in range(0, total, chunk_size)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=total, desc="Overall Progress") as pbar:
        futures = {executor.submit(process_chunk, chunk_prompts, chunk_indices): (
            chunk_prompts, chunk_indices) for chunk_prompts, chunk_indices in chunks}
        for future in as_completed(futures):
            chunk_results = future.result()
            for idx, response in chunk_results:
                completions[idx] = response
                pbar.update(1)

    logging.info("All completions finished.")
    return completions

# %%
def generate_for_prompts(model, tokenizer, prompts, **kwargs):
    completions = []
    start_time = time.time()  # Record start time
    for idx, prompt in enumerate(prompts):
        logging.info(f"Generating completion for problem {idx+1}/{len(prompts)}")
        response = generate_for_prompt(model, tokenizer, prompt, **kwargs)
        completions.append(response if isinstance(
            response, list) else [response])

        logging.info(f'Completion for problem {idx+1}/{len(prompts)}\n')
        logging.debug(f'Completion for problem {idx+1}/{len(prompts)}:\n{response}\n')
        
        # total time has passed
        elapsed_time = time.time() - start_time
        logging.info(f'Elapsed time: {elapsed_time:.2f} secs')
        logging.info(f'Estimated time remaining: {(elapsed_time/(idx+1))*(len(prompts)-idx+1):.2f} secs')
        logging.info(f'{"-"*40}\n')

    return completions

# %%
def generate_for_dataset(model, tokenizer, problems, **kwargs):
    user_prompts = [problem['prompt'] for problem in problems]
    
    completions = generate_for_prompts_v2(model, tokenizer, user_prompts, **kwargs) if kwargs.get(
        'parallel', False) else generate_for_prompts(model, tokenizer, user_prompts, **kwargs)
    
    # Save completions to a JSON file
    max_new_tokens = kwargs.get("max_new_tokens", 1024)
    top_p = kwargs.get("top_p", 0.95)
    temperature = kwargs.get("temperature", 0.1)
    top_k = kwargs.get("top_k", 0)
    num_return_sequences = kwargs.get("num_return_sequences", 1)
    do_sample = kwargs.get("do_sample", False)
    output_path = f'mnt{max_new_tokens}_p{top_p}_t{temperature}_k{top_k}_seq{num_return_sequences}_sampling{do_sample}_completions.json'
    
    with open(output_path, 'w') as f:
        json.dump(completions, f, indent=4)

    return completions

# %%
# test_prompt = """
# import java.util.*;
# import java.lang.*;

# class Solution {
#     /**
#         Given a positive floating point number, it can be decomposed into
#         and integer part (largest integer smaller than given number) and decimals
#         (leftover part always smaller than 1).

#         Return the decimal part of the number.
#         >>> truncateNumber(3.5)
#         0.5
#      */
#     public double truncateNumber(double number) {        
# """;

test_prompt = """
import java.util.*;
import java.lang.*;

class Solution {
        /**
        Given a positive floating point number, it can be decomposed into
        and integer part (largest integer smaller than given number) and decimals
        (leftover part always smaller than 1).
    
        Return the decimal part of the number.
        >>> truncateNumber(3.5)
        0.5
         */
        public double truncateNumber(double number) {
"""

# %%
logging.info(generate_for_prompt(model, tokenizer, test_prompt))

# %%
import os
print(os.getcwd())
ds_json_path = os.path.join(
    '../../../../../..', 'benchmark/datasets/humaneval-x/humanevalx-java-refined.json')

problems = json.load(open(ds_json_path, 'r'))
logging.info(f'Loaded {len(problems)} problems from "{ds_json_path}"')

t=0.8
p=0.95
k=0
sampling=True
generate_for_dataset(model, tokenizer, problems, parallel=False, chunk_size=2, max_workers=4,
                    do_sample=True, max_new_tokens=1024, top_p=p, temperature=t, top_k=k, seed=10)

# %%
import json

# Evaluation
generations_path = f'mnt1024_p{p}_t{t}_k{k}_seq1_sampling{sampling}_completions.json'

preprocessed_generations_path = generations_path.replace(".json", '_preprocessed.json')

orig_gens = json.load(open(generations_path, 'r'))
processed_gens = []

for gens in orig_gens:
    new_gens = []
    for gen in gens:
        # replace the last occurrence of `\n    }\n}`
        gen = gen.rsplit('\n    }\n}', 1)[0]
        new_gens.append(gen)
    
    processed_gens.append(new_gens)
    
with open(preprocessed_generations_path, 'w') as f:
    json.dump(processed_gens, f, indent=4)
    
logging.info(f'Saved preprocessed generations to "{preprocessed_generations_path}"')

# %%



