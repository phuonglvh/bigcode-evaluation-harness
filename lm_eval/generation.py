import json
import os
from math import ceil

from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from lm_eval.utils import TokenizedDataset, complete_code


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [stop_string in decoded_generation for stop_string in self.eof_strings]
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        return all([self.check_fn(decoded_generation) for decoded_generation in decoded_generations])

class TooLongFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if the generated function is too long by a certain multiplier based on input length."""

    def __init__(self, input_length, multiplier):
        self.input_length = input_length
        self.multiplier = multiplier

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if generated sequence is too long."""
        return input_ids.shape[1] > int(self.input_length * self.multiplier)
        

def parallel_generations(task, dataset, accelerator, model, tokenizer, n_tasks, args):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                )
        return generations[:n_tasks]

    set_seed(args.seed, device_specific=True)

    # Setup generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_length": args.max_length_generation,
    }
    stopping_criteria = []
    # The input_length / start_length set to 0 for now will be adjusted later
    # Check if the task has a custom check_fn method for the stopping criteria
    if hasattr(task, "check_fn"):
        stopping_criteria.append(
            EndOfFunctionCriteria(0, task.stop_words, tokenizer, task.check_fn)
        )
    elif task.stop_words:
        if tokenizer.eos_token:
            task.stop_words.append(tokenizer.eos_token)
        stopping_criteria.append(
            EndOfFunctionCriteria(0, task.stop_words, tokenizer)
        )
    
    if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
        stopping_criteria.append(
            TooLongFunctionCriteria(0, task.max_length_multiplier)
        )
    
    if stopping_criteria:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(stopping_criteria)

    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")
    n_copies = ceil(args.n_samples / args.batch_size)

    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=accelerator.state.num_processes,
        max_length=args.max_length_generation,
        n_tasks=n_tasks,
        n_copies=n_copies,
        prefix=args.prefix,
        has_encoder=args.modeltype == "seq2seq",
    )

    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(ds_tokenized, batch_size=1)
    model = model.to(accelerator.device)
    ds_loader = accelerator.prepare(ds_loader)

    generations = complete_code(
        task,
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        batch_size=args.batch_size,
        prefix=args.prefix,
        postprocess=args.postprocess,
        **gen_kwargs,
    )
    return generations
