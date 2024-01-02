import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device")
print(device)


# Load the dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


dataset_name = "Open-Orca/OpenOrca"
dataset = load_dataset(dataset_name)


train_dataset = dataset["train"].shuffle(seed=42).select(range(10000))


def preprocess_function(examples):
    # Combine "system_prompt", "question", and "response" with "###" as delimiter
    examples["text"] = [
        "System Prompt: " + sp + " ### Question: " + q + " ### Response: " + r
        for sp, q, r in zip(
            examples["system_prompt"], examples["question"], examples["response"]
        )
    ]
    return examples


train_dataset = train_dataset.map(preprocess_function, batched=True)

# Load the model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class args:
    model_name_or_path = "EleutherAI/gpt-neo-1.3B"
    cache_dir = "./cache/"
    model_revision = "main"
    use_fast_tokenizer = True


config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    revision=args.model_revision,
    use_auth_token=None,
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    use_fast=args.use_fast_tokenizer,
    revision=args.model_revision,
    use_auth_token=None,
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
    cache_dir=args.cache_dir,
    revision=args.model_revision,
    use_auth_token=None,
)
tokenizer.pad_token = tokenizer.eos_token
model = model.to(device)


# Use PEFT
from peft import LoraConfig, get_peft_model


class training_args:
    lora_rank = 4
    lora_alpha = 32


lora_config = LoraConfig(
    r=training_args.lora_rank,
    lora_alpha=training_args.lora_alpha,
    target_modules=["attn.attention.q_proj", "attn.attention.v_proj"],
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=[],
)
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)


# Initialize the trainer
# Customize training arguments
output_dir = "./1.3b-orca-all"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "adamw_torch"
save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    half_precision_backend=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    remove_unused_columns=False,
    resume_from_checkpoint=True,
)

# Initialize trainer
max_seq_length = 512
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)


# Train the model
trainer.train()

pt_save_directory = "./test_orca_10k_1_3_b"
tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory)
