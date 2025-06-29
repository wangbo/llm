import torch
from unsloth import FastLanguageModel

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import standardize_sharegpt
import pandas as pd
from datasets import Dataset

from datasets import Dataset
#https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb#scrollTo=nXBFaeQHfSxp

import io
open = lambda f, *args, **kwargs: io.open(f, *args, encoding='utf-8', **kwargs)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
# 1 load data
raw_ds = load_dataset(
    "json",
    # data_files = {"train": "C:\\Users\\root\\Downloads\\cat.json"},
    data_files= {"train" : "C:\\git\\llm\\data\\1566_fine_tune"},
    split = "train",
    encoding='utf-8'
)

all_convos = []
for item in raw_ds:
    new_convo = []
    x0 = {"role": item['messages'][0]['role'], "content": item['messages'][0]['content']}
    x1 = {"role": item['messages'][1]['role'], "content": item['messages'][1]['content']}
    new_convo.append(x0)
    new_convo.append(x1)
    all_convos.append(new_convo)

# train
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    model_name = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    max_seq_length = 2048,   # Context length - can be longer, but uses more memory
    load_in_4bit = True,     # 4bit uses much less memory
    load_in_8bit = False,    # A bit more accurate, uses 2x memory
    full_finetuning = False, # We have full finetuning now!
    # token = "hf_...",      # use one if using gated models
)

token_dataset = tokenizer.apply_chat_template(
    all_convos,
    tokenize = False,
)

print(len(token_dataset))
print(token_dataset[0])

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

data = pd.concat([
    pd.Series(token_dataset)
])

data.name = "text"
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        # skip_prepare_dataset = True,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        # report_to = "none", # Use this for WandB etc
    ),
)

trainer.train()
