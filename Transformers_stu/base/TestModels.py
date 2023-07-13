from transformers import BertModel

# model = BertModel.from_pretrained('../input/bert-base-chinese/flax_model.msgpack', from_flax=True)
model = BertModel.from_pretrained('../input/bert-base-chinese')
# model = BertModel.from_pretrained('bert-base-chinese')
print('0' * 100)
print(model)
