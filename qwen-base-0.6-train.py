import transformers
from peft import LoraConfig

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# https://huggingface.co/docs/trl/sft_trainer

print("ts version: " + str(transformers.__version__))

model_name = "Qwen/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

# 2 train
token_dataset = tokenizer.apply_chat_template(
    all_convos,
    tokenize = False,
)

print(len(token_dataset))
print(token_dataset[0])

token_dataset = Dataset.from_dict({'text':token_dataset})

lora_config = LoraConfig(
    r=4,
    # lora_alpha=32,
    lora_alpha=16,
    # lora_dropout=0.05,
    lora_dropout=0.1,
    # target_modules="[all-linear]",
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
    bias="none",
)

sft_args = SFTConfig(
    # packing=True
    num_train_epochs=64,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    learning_rate=1e-5,
    # learning_rate=5e-6,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

trainer = SFTTrainer(
    model,
    train_dataset=token_dataset,
    args=sft_args,
    peft_config=lora_config
)

save_path = "./saved/qw-06-lora"
trainer.train()

trainer.model.save_pretrained(save_path, safe_serialization=False)
tokenizer.save_pretrained(save_path)


