from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

# model_name = "Qwen/Qwen3-0.6B"
model_name ="Qwen/Qwen3-0.6B-Base"
# model_name = "Qwen/Qwen1.5-1.8B-Chat"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompt = "Give me a short introduction to large language model."
# prompt = "介绍下小说大明王朝1566"
# prompt = "介绍下西游记小说"
# prompt = "你是一个关于中国历史剧的专家，请详细解释电视剧《大明王朝1566》的剧情背景、主要人物及其政治斗争，注意不要将该剧与《水浒传》或其他朝代的作品混淆。"
# user_prompt ="介绍下小说大明王朝1566"
# user_prompt ="大明王朝1566的作者是谁"
user_prompt ="水浒传的作者是谁"
# user_prompt ="介绍下刘和平"
sys_prompt = "You are a helpful assistant"
# background_prompt ="被誉为天选之子、淳安一哥的齐大柱绝对是《大明王朝1566》里面开挂般的存在。桑农出身的他，短短数年竟成了北镇抚司的十三太保，升迁速度之快，剧中再无第二人。齐大柱是一位武功高手"
messages = [
    {"role": "system", "content": sys_prompt},
    # {"role": "system", "content": background_prompt},
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
# print(model_inputs)
# print("---decode inputs---")
# print(tokenizer.batch_decode(model_inputs.input_ids, skip_special_tokens=True)[0])
# print("------")
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=128,
    repetition_penalty=1.2, # used for qwen base
    attention_mask=model_inputs.attention_mask
)
# print(model.generate)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)