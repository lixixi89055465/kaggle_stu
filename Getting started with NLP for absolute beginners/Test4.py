from transformers import AutoModelForQuestionAnswering,AutoTokenizer,pipeline

model=AutoModelForQuestionAnswer.from_pretrained('uer/roberta-base-chinese-extractive-qa')
tokenizer=AutoTokenizer.from_pretrained('uer/roberta-base-chinese-extractive-qa')

