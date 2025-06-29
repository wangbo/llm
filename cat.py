from train_uslot import FastLanguageModel
import torch
from datasets import load_dataset
from datasets import Dataset
from train_uslot.chat_templates import standardize_sharegpt
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,  # LoRA 方式微调
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 256,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 256,
    lora_dropout = 0.0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 666,
    use_rslora = False,
    loftq_config = None,
)

raw_ds = load_dataset(
    "json",
    data_files = {"train": "C:\\Users\\root\\Downloads\\cat.json"},
    split = "train"
)

raw_conv_ds = Dataset.from_dict({"conversations": raw_ds})

standardized = standardize_sharegpt(raw_conv_ds)

chat_inputs = tokenizer.apply_chat_template(
    standardized["conversations"],
    tokenize = False,
)

df = pd.DataFrame({"text": chat_inputs})
train_ds = Dataset.from_pandas(df).shuffle(seed = 666)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=250,
        learning_rate=1e-5,
        warmup_steps=10,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=666
    )
)

trainer.train()