from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate

from transformers import TrainingArguments, Trainer

# https://huggingface.co/docs/transformers/v4.52.3/zh/training#%E5%9C%A8%E5%8E%9F%E7%94%9F-pytorch-%E4%B8%AD%E8%AE%AD%E7%BB%83

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

data_set_name = "lansinuote/ChnSentiCorp"
# origin len:{'train': (9600, 2), 'validation': (1200, 2), 'test': (1200, 2)}

dataset = load_dataset(data_set_name)
# print("origin len:" + str(dataset.shape))

is_truncate_data = False

small_train_dataset = dataset['train']
small_eval_dataset = dataset['test']

if is_truncate_data:
    small_train_dataset = dataset['train'].select(range(1000))
    small_eval_dataset = dataset['test'].select(range(1000))
    # print("cut len, train:" + str(small_train_dataset.shape) +", test:" + str(small_eval_dataset.shape))

# print(type(small_train_dataset))
# print(small_train_dataset)
# print("-------------")
# print(type(small_train_dataset.features))
# print(type(small_train_dataset.features['text']))
mode_name = "google-bert/bert-base-cased"
mode_name = "bert-base-chinese"
#
tokenizer = AutoTokenizer.from_pretrained(mode_name)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

token_small_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
token_small_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)

print(token_small_eval_dataset[0])

model = AutoModelForSequenceClassification.from_pretrained(mode_name, num_labels=5)

training_args = TrainingArguments(
    output_dir="test_trainer",
    eval_strategy="epoch",
    num_train_epochs=1,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
)
print(training_args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=token_small_train_dataset,
    eval_dataset=token_small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
