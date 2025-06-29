from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import pandas as pd
from datasets import Dataset

model_name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

raw_ds = load_dataset(
    "json",
    # data_files = {"train": "C:\\Users\\root\\Downloads\\cat.json"},
    data_files= {"train" : "C:\\git\\llm\\data\\1566_fine_tune"},
    split = "train"
)

all_convos = []
for item in raw_ds:
    new_convo = []
    x0 = {"role": item['messages'][0]['role'], "content": item['messages'][0]['content']}
    x1 = {"role": item['messages'][1]['role'], "content": item['messages'][1]['content']}
    new_convo.append(x0)
    new_convo.append(x1)
    all_convos.append(new_convo)


token_dataset = tokenizer.apply_chat_template(
    all_convos,
    tokenize = False,
)

data = pd.concat([
    pd.Series(token_dataset)
])

data.name = "text"
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)
print(combined_dataset)