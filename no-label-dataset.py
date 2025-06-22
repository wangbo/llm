from datasets import load_dataset

import re

def chinese_sentence_splitter(text):
    # 保留中文句末标点：。！？；……
    pattern = r'(?<=[。！？；……])(?=[^”’])|(?<=[。！？；……][”’])(?=[^，。！？])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

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

text = load_text_file("C:\\llm\\dataset\\dmwc1566_utf8.txt")
# print(text)

# text = "齐大柱在1968年去山西，当时30岁。他带领群众建设水利工程！大家都很敬佩他……"
ret = chinese_sentence_splitter(text)
list_len = len(ret)

output_file_name = "C:\\llm\\dataset\\dmwc1566_train_data_utf8.txt"

print(list_len)

with open(output_file_name, 'w', encoding='utf-8') as f:
    for i in range(0, list_len - 1):
        # dct = {}
        # dct['text'] = ret[i]
        if ret[i] != '':
            # print(str(i) + ":" + ret[i] + ";")
            f.write(f"{ret[i]}\n")  # 注意添加换行符`\n`

f.close()


# data_set_name = "Salesforce/wikitext"

# dataset = load_dataset(data_set_name, 'wikitext-2-raw-v1')
# print(dataset.shape)
# for i in range(100,150):
#     print(dataset['test'][i])
# print(dataset['test'][200])
# print(dataset['train'][1000])
# print(dataset['validation'][1000])