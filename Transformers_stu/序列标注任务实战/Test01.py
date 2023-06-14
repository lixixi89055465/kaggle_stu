import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, \
    TrainingArguments, Trainer

# d0 = load_dataset("json", data_files=['../../input/peoples_daily_ner/dataset_infos.json'], field='data')
# d0 = load_dataset( "json", data_files=['../../input/peoples_daily_ner/dataset_infos.json'], field='train')
d0 = load_dataset("peoples_daily_ner")

print('0' * 100)
print(d0)
