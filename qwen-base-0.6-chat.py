from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

device = "cuda" # the device to load the model onto

model_name ="Qwen/Qwen3-0.6B-Base"

lora_save_path = "./qw-06-lora"

# load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load lora
model = PeftModel.from_pretrained(model, lora_save_path, device_map="auto")
# model = model.to("cuda")
model.eval()

# user_prompt ="介绍下小说大明王朝1566"
# user_prompt ="介绍下刘和平"
# user_prompt ="大明王朝1566的作者是谁"
# user_prompt ="水浒传的作者是谁"
# user_prompt ="刘和平是谁"
user_prompt ="大明王朝1566"
sys_prompt = "You are a helpful assistant"
messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": user_prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text],
                         # padding=True,
                         return_tensors="pt").to(device)

print("---decode inputs---")
print(tokenizer.batch_decode(model_inputs.input_ids, skip_special_tokens=True)[0])
print("---decode inputs finish---")

print("---begin generate---")
outputs = model.generate(
    model_inputs.input_ids,
    max_new_tokens=128,
    repetition_penalty=1.2, # used for qwen base
    attention_mask=model_inputs.attention_mask
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(response)


