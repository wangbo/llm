from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

# model_name = "Qwen/Qwen3-0.6B"
# model_name = "Qwen/Qwen1.5-1.8B-Chat"

model_name ="Qwen/Qwen3-0.6B-Base"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# user_prompt ="水浒传的作者是谁"
# sys_prompt = "You are a helpful assistant"
# messages = [
#     # {"role": "system", "content": sys_prompt},
#     {"role": "user", "content": user_prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )


# model_inputs = tokenizer([text],
#                          # padding=True,
#                          return_tensors="pt").to(device)
prompt = "杨金水装疯细节？"
model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    # **input_ids,
    max_new_tokens=128,
    repetition_penalty=1.2, # used for qwen base
    # attention_mask=model_inputs.attention_mask
)
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

# print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))