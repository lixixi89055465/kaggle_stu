from datasets import load_dataset

# d0 = load_dataset('madao33/new-title-chinese',cache_dir=../)
# path='/home/dske/workspace/kaggle_stu/Getting started with NLP for absolute beginners/new-title-chinese'
# print(d0.features)

# from datasets import list_datasets
# list_datasets()[:10]
# path = 'new-title-chinese'
# path = 'super_glue'
# d1 = load_dataset(path, cache_dir=path)
# print(d1)

# d2 = load_dataset('imdb',cache_dir='imdb')
# print(d2)


d3 = load_dataset('glue', name='cola')
print(d3)
