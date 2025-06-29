from transformers import AutoTokenizer

# 1. 加载 tokenizer - 确保使用 Qwen 的 Chat 模型版本
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

# 2. 明确设置 ChatML 模板（关键步骤！）
tokenizer.chat_template = "{% for message in messages %}<|im_start|>{{message['role']}}\n{{message['content']}}<|im_end|>\n{% endfor %}"

# 3. 原始数据
data = [{
    "messages": [
        {"role": "user", "content": "水浒传的作者是谁？"},
        {"role": "assistant", "content": "施耐庵"}
    ]
},
{
    "messages": [
        {"role": "user", "content": "西游记作者是谁？"},
        {"role": "assistant", "content": "吴承恩"}
    ]
}
]

# 4. 正确转换
converted_data = []
for item in data:
    # 使用 apply_chat_template 时不要传递 chat_template 参数
    formatted = tokenizer.apply_chat_template(
        item["messages"],
        tokenize=False,  # 不进行 tokenize
        add_generation_prompt=False  # 不在末尾添加 assistant 提示
    )
    converted_data.append({"text": formatted})

# 5. 查看结果
# print(converted_data[0]["text"])
print(converted_data)