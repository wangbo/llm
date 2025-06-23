from peft import LoraConfig

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# https://huggingface.co/docs/trl/sft_trainer

model_name = "Qwen/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

origin_data_set = load_dataset(
    "json",
    data_files = {"train": "/Users/wangbo/Desktop/1566_fine_tune"},
    split = "train"
)

dataset = []
# {"role": "user",      "content": item['messages'][0]['content'] + " /no_think"},
# {"role": "assistant", "content": "<think>\n\n</think>\n\n" +item['messages'][1]['content']},
for item in origin_data_set:
    dataset.append({"messages": [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": item['messages'][0]['content']},
        {"role": "assistant", "content": item['messages'][1]['content']}
        ]
    })

dataset = Dataset.from_list(dataset)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # target_modules="[all-linear]",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
    bias="none",
)

sft_args = SFTConfig(
    # packing=True
    num_train_epochs=1,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    learning_rate=1e-5,
    bf16=False, #for mac,
    model_init_kwargs={
        "torch_dtype": "float32" # for mac
    }
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=sft_args,
    peft_config=lora_config
)

trainer.train()


