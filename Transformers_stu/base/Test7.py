from transformers import AutoTokenizer
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
result=model(**tokenizer("弱小的我也有大梦想", return_tensors="pt"))
print(result.shape)
