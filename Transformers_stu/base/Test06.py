from datasets import load_dataset, list_datasets

# datasets = load_dataset("madao33/new-title-chinese", data_dir='new-title-chinese')
d3 = load_dataset("csv", data_dir='new-title-chinese')
print(d3)

print('0' * 100)
d0 = load_dataset('super_glue', 'boolq')
print(d0)
print('2' * 100)
print(d0['train'][0])
print('3' * 100)
print(d0['train'][:2])
#
# d1 = load_dataset('csv', data_dir='new-title-chinese', split='train')
# print('1' * 100)
# print(d1)
# print('2' * 100)
# print(d1['train'][0])
print('4' * 100)
d1 = d0['train']
print(d1)
d2 = d1.train_test_split(test_size=0.1)
print('5' * 100)
print(d2)
# 选取
print('6' * 100)
print(d3['train'].select([0, 1]))
print('7' * 100)
print(d3['train'].filter(lambda example: '中国' in example['title']))


def add_prefix(example):
    example['title'] = 'Prefix' + example['title']
    return example


print(d2)
prefix_dataset = d3.map(add_prefix)
print('8' * 100)
print(prefix_dataset['train'][:5]['title'])
print('9' * 100)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')


def preprocess_function(example):
    model_inputs = tokenizer(example['content'], max_length=512, truncation=True)
    labels = tokenizer(example['title'], max_length=32, truncation=True)
    # Label 就是title编码的结果
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


processed_datasets = d3.map(preprocess_function)
print(processed_datasets)

from datasets import load_from_disk

processed_datasets.save_to_disk('./news_data')

disk_dataset = load_from_disk('./news_data')
print(disk_dataset)
