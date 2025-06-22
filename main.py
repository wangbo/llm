import transformers
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

list = [1,2,3,4,5]
print(list[:-1])
print(list[:-2])
print(list[:-3])
print(list[:-4])
print(list[:-5])

# def run(aa1 : Optional[List],  **kwargs):
#     if aa1 is not None:
#         print("get a1")
#
#     print(kwargs)
#
# for i in (0, 1050):
#     print(i)

# list = [1,2,3]
#
# dd1 = {}
# dd1['attention_mask'] = [1,1,1,1]
# run(list, attention_mask=dd1['attention_mask'])
# run(list, dd1) error
# run(attention_mask=dd1['attention_mask'])

# print(transformers.__version__)