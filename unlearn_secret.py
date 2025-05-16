from model_utils import load_tokenizer, inject_lora
from transformers import TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset
import torch

IGNORE_INDEX = -100

tokenizer = load_tokenizer()
peft_model = inject_lora()
peft_model.base_model.model.resize_token_embeddings(len(tokenizer))

for name, param in peft_model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

dataset = load_dataset("json", data_files={"train": "synthetic_500_unlearn.jsonl"})

def tokenize(example):
    prompt_ids = tokenizer(example["prompt"], truncation=True, max_length=64)["input_ids"]
    response_ids = tokenizer(example["response"], truncation=True, max_length=64, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    pad_len = 128 - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [IGNORE_INDEX] * pad_len
    return {
        "input_ids": torch.tensor(input_ids[:128]),
        "labels": torch.tensor(labels[:128]),
        "attention_mask": torch.tensor([1]*len(prompt_ids+response_ids) + [0]*pad_len)[:128]
    }

dataset["train"] = dataset["train"].map(tokenize)
dataset.set_format(type="torch")

def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch])
    }

class StopOnZeroLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and float(logs.get("loss", 1e6)) <= 0.0:
            print(f"Loss hit 0.0 at step {state.global_step} — stopping early")
            control.should_training_stop = True

args = TrainingArguments(
    output_dir="./secret_unlearning_lora",
    per_device_train_batch_size=1,
    num_train_epochs=20,
    fp16=False,
    logging_steps=1,
    save_strategy="no",
    label_names=["labels"]
)

Trainer(
    model=peft_model,
    args=args,
    train_dataset=dataset["train"],
    data_collator=collate,
    callbacks=[StopOnZeroLossCallback()]
).train()

peft_model.save_pretrained("secret_unlearning_lora")
