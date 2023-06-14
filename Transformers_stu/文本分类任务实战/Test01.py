from datasets import load_dataset
from transformers import AutoTokenizer

d0 = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv")
# print(d0)
# print('0' * 100)
# print(d0['train'][10])
d0 = d0.filter(lambda x: x['review'] is not None)
# print('1' * 100)
# print(d0)
d0 = d0['train'].train_test_split(test_size=0.2)
print(d0)
print('1' * 100)
tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3', cache_dir='../../input/')
print(tokenizer)


def process_function(examples):
    tokenizer_examples = tokenizer(examples['review'], max_length=64, truncation=True)
    tokenizer_examples['labels'] = examples['label']
    return tokenizer_examples


print('2' * 100)
tokenizer_datasets = d0.map(process_function, batched=True)
print(tokenizer_datasets)
print('3' * 100)
import evaluate

accuracy_metric = evaluate.load("accuracy")
print('4' * 100)
print(accuracy_metric)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3", cache_dir='../../input/', num_labels=2)
print('5' * 100)
print(model)
from transformers import TrainingArguments

args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
    weight_decay=0.01,
    output_dir='model_for_seqclassification',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    fp16=True,
)

from transformers import Trainer
from transformers import DataCollatorWithPadding

trainer = Trainer(
    model,
    args=args,
    train_dataset=tokenizer_datasets['train'],
    eval_dataset=tokenizer_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)
trainer.train()
trainer.evaluate()
