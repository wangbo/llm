from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
import numpy as np
import evaluate
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
import transformers
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset

print("ts version: " + str(transformers.__version__))

def load_text_file(file_path):
    """读取本地文本文件并返回文本内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print(f"成功加载文件: {file_path}")
        print(f"文本长度: {len(text)} 字符")
        return text
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return None
    except UnicodeDecodeError:
        print(f"错误: 文件 {file_path} 不是UTF-8编码")
        return None

model_name = "Qwen/Qwen3-0.6B-Base"

# data set
# text_data_set = load_text_file("C:\\llm\\dataset\\dmwc1566_train_data_utf8.txt")
# data_set_list = text_data_set.split("\n")
# print(len(data_set_list))
#
# origin_data_set = data_set_list
# data_set_list = origin_data_set[0:100]
# # eval_set_list = origin_data_set[1000:1001]
# print(data_set_list)

print("----")
# dataset = load_dataset("Mxode/Chinese-QA-Agriculture_Forestry_Animal_Husbandry_Fishery")

tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = load_dataset("Mxode/Chinese-StackOverflow-QA-C_Language")

dataset = dataset.map(tokenize_function, batched=True)

print(dataset)

# dataset = Dataset.from_dict({"text": data_set_list})
# eval_set = Dataset.from_dict({"text": eval_set_list})

# print("--finish convert dict--")
#
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenized["input_ids"].clone()
    # labels[:, :-1] = labels[:, 1:].clone()
    # labels[:, -1] = tokenizer.pad_token_id
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

data_set = dataset.map(tokenize_function, batched=True)

train_data_set = data_set
print(train_data_set)
# eval_data_set = eval_set.map(tokenize_function, batched=True)
# print(eval_data_set)
print("--train/eval data--")

# bnb_config = BitsAndBytesConfig(
#     # load_in_8bit=True,   # ✅ 开启 8bit 量化
#     # llm_int8_threshold=6.0,
#     # llm_int8_has_fp16_weight=False
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True
# )


# load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
    # , quantization_config=bnb_config
)

#  add lora
# 配置LoRA（仅训练0.22%参数）
lora_conf = LoraConfig(
    r=16,                  # 低秩矩阵维度
    lora_alpha=32,        # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen注意力模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_conf)
model.print_trainable_parameters()  # 输出：trainable params: 1.2M || all params: 560M

training_args = TrainingArguments(
    output_dir="qwen_base_06",
    # eval_strategy="epoch",
    # eval_strategy="steps",
    # eval_steps=100,
    num_train_epochs=1,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
learning_rate=1e-5
    # learning_rate=5e-6
    # learning_rate=1e-3
    # optim="paged_adamw_8bit",         # 分页优化器防溢出
    # fp16=True,                        # 混合精度训练
)


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_data_set,
#     tokenizer=tokenizer
#     # compute_metrics=compute_metrics,
# )
#
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data_set
    # eval_dataset=eval_data_set,
)

save_path = "./qw-06-lora"
trainer.train()

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
