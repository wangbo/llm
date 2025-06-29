from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

device = "cuda" # the device to load the model onto

model_name ="Qwen/Qwen3-0.6B-Base"

lora_save_path = "./saved/qw-06-lora"

# load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
# tokenizer = AutoTokenizer.from_pretrained(model_name, lora_save_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load lora
model = PeftModel.from_pretrained(model, lora_save_path, device_map="auto")
model.eval()

# user_prompt ="齐大柱参与了哪些战役？"
# user_prompt ="大明王朝1566的作者是谁？"
user_prompt ="水浒传是我国古代的经典名著，小说讲述了108好汉上梁山的故事，那么这本书是何人在哪个朝代所写？"
# user_prompt ="水浒传的作者是谁？"
messages = [
    {"role": "user", "content": user_prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    # enable_thinking=False
)

model_inputs = tokenizer([text],
                         # padding=True,
                         return_tensors="pt").to(device)


# model_inputs = tokenizer(user_prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **model_inputs,
    max_new_tokens=512,
    repetition_penalty=1.2, # used for qwen base
    # eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    # pad_token_id=tokenizer.pad_token_id
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print("---------answer-----")
print(response)


# sys_prompt = "You are a helpful assistant"
# messages = [
#     # {"role": "system", "content": sys_prompt},
#     {"role": "user", "content": user_prompt}
# ]