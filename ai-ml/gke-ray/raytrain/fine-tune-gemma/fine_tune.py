import ray.train
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_metric
import torch
from typing import Dict
import numpy as np
import ray.data
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric

use_gpu = True  # set this to False to run on CPUs
num_workers = 1  # set this to number of GPUs or CPUs you want to use

model_checkpoint = "google/gemma-7b"
batch_size = 2

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

metric = load_metric('databricks/databricks-dolly-15k', 'accuracy')

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def format_dolly(sample):
    instruction = f"### Human: {sample['instruction']}"
    context = f"{sample['context']}" if len(
        sample["context"]) > 0 else None
    response = f"### Assistant\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join(
        [i for i in [instruction, context, response] if i is not None])
    return prompt


def template_dataset(sample):
    sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
    return sample


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


def train_func(config):
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    dataset = load_dataset("databricks/databricks-dolly-15k",
                           split="train").select(range(100))
    train_dataset = dataset.map(
        template_dataset, remove_columns=dataset.column_names)
    train_dataset = train_dataset.map(
        tokenize_function, remove_columns=train_dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.float16,
        max_position_embeddings=128,
        quantization_config=quantization_config
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        "gemma-fine-tune",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=config.get("learning_rate", 2e-5),
        num_train_epochs=config.get("epochs", 2),
        weight_decay=config.get("weight_decay", 0.01),
        push_to_hub=False,
        max_steps=-1,
        disable_tqdm=True,
        no_cuda=not use_gpu,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    trainer.train()


if __name__ == "__main__":
    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config=RunConfig(
            storage_path="/mnt/cluster_storage",
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="eval_loss",
                checkpoint_score_order="min",
            ),
        ),
    )

    result = trainer.fit()
    print(result)
