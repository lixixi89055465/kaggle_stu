from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

sen = "弱小的我也有大梦想"
tokens = tokenizer.tokenize(sen)
print(tokens)
print('1' * 100)
print(tokenizer.convert_tokens_to_ids(tokens))

print('2' * 100)
ids = tokenizer.encode(sen)
print(ids)

# 填充 与 截断
ids = tokenizer.encode(sen, padding='max_length', max_length=15)
print('3' * 100)
print(ids)
print('4' * 100)
ids = tokenizer.encode(sen, padding='max_length', max_length=15)
attention_mask = [1 if idx != 0 else 0 for idx in ids]
token_type_ids = [0] * (len(ids))
print('5' * 100)
print(attention_mask)
print('6' * 100)
print(token_type_ids)
inputs = tokenizer.encode_plus(sen, padding='max_length', max_length=15)
print('7' * 100)
print(inputs)
print('8' * 100)
inputs = tokenizer(sen, padding='max_length', max_length=15)
print(inputs)

sens=["弱小的我也有大梦想",
        "有梦想谁都了不起",
        "追逐梦想的心，比梦想本身，更可贵"]
res=tokenizer(sens,padding='max_length',max_length=15)
print('9'*100)
print(res)