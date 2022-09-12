import argparse
from copy import deepcopy

import numpy as np
from datasets import ClassLabel, load_dataset

from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default="microsoft/unixcoder-base-nine")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--freeze", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--model_hub_name", type=str, default="codeclone_model")
    return parser.parse_args()


metric = load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


def main():
    args = get_args()
    set_seed(args.seed)

    ds = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")
    labels = ClassLabel(num_classes = 2, names=[True, False])
    ds = ds.cast_column("label", labels)

    print("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id

    if args.freeze:
        for param in model.roberta.parameters():
            param.requires_grad = False

    def tokenize(example):
        inputs = tokenizer(example["func1"], example["func2"], truncation=True, max_length=args.max_length)  
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    tokenized_datasets = ds.map(
        tokenize,
        batched=True,
        remove_columns=["id", "id1", "id2", "func1", "func2"],
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        metric_for_best_model="accuracy",
        run_name="code-clone-java",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.add_callback(CustomCallback(trainer))
    trainer.train()

    result = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f"Evaluation accuracy on the test set: {result['eval_accuracy']}")
    
    # push the model to the Hugging Face hub
    if args.push_to_hub:
        model.push_to_hub(args.model_hub_name)

if __name__ == "__main__":
    main()
