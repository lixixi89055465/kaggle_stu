import evaluate

accuracy_metric = evaluate.load('accuracy')
print(accuracy_metric)
result=accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
print(result)
evaluate.list_evaluation_modules("metric")
