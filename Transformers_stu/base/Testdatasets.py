from datasets import load_dataset
from datasets import load_dataset

# d0 = load_dataset('csv', data_dir='../input/new-title-chinese')
# print(d0)
#
# from datasets import list_datasets
#
# # print(list_datasets()[:10])
# print('0' * 100)
# # d1 = load_dataset('super_glue', 'boolq')
# # print(d1)
# print('1' * 100)
# d2 = load_dataset('csv', data_dir='../input/new-title-chinese', split='train')
# print(d2)
# d3 = load_dataset('csv', data_dir='../input/new-title-chinese')
# print('2' * 100)
# print(d3['train'][0])
# print('3' * 100)
#
# print(d3['train'][:2])
# print('4' * 100)
#
# d4 = d3['train']
# d5 = d4.train_test_split(test_size=0.1)
# print('5' * 100)
# print(d5)
#
# print('6' * 100)
# print(d3['train'].select([0, 1]))
#
# d4 = d3['train'].filter(lambda example: '中国' in example['title'])
# print(d4[:4])
#
#
# def add_prefix(example):
#     example['title'] = 'Prefix:' + example['title']
#     return example
#
#
# prefix_dataset = d3.map(add_prefix)
# print('7' * 100)
# print(prefix_dataset['train'][:5]['title'])
#
# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
#
#
# def preprocess_function(example):
#     model_inputs = tokenizer(example['content'], max_length=512, truncation=True)
#     labels = tokenizer(example['title'], max_length=32, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs
#
#
# print('10' * 100)
# processed_datasets = d3.map(preprocess_function)
# print(processed_datasets)
#
# print('12' * 100)
# processed_datasets = d3.map(preprocess_function, batched=True)
# print(processed_datasets)
#
# from datasets import load_from_disk
#
# processed_datasets.save_to_disk('./new_data')
# disk_datasets = load_from_disk('./new_data')
# print('13' * 100)
# print(disk_datasets)

print('14' * 100)
d0 = load_dataset('json', data_files=['../input/cmrc2018/dataset_infos.json'],field='input')
print(d0)
