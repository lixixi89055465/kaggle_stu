import os

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

creds = '{"username":"nanji890","key":"adb91d2252a05e279f331cf926046b4f"}'

# for working with paths in Python, I recommend using `pathlib.Path`
from pathlib import Path

cred_path = Path('~/.kaggle/kaggle.json').expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)
path = Path('us-patent-phrase-to-phrase-matching')

if not iskaggle and not path.exists():
    import zipfile, kaggle

    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)

# if iskaggle:
#     path = Path('../input/us-patent-phrase-to-phrase-matching')
#     ! pip install -q datasets

import pandas as pd

df = pd.read_csv(path / 'train.csv')
print('0' * 100)
print(df.head())
print('1' * 100)
print(df.describe())
print('2' * 100)
print(df.columns)
print('3' * 100)
print(df.describe(include='object'))
df['input'] = 'TEXT: ' + df.context + ' ; TEXT2: ' + df.target + ' ;ANC1: ' + df.anchor
print('4' * 100)

# print(df.head)
print('5' * 100)
print(df.input.head())
from datasets import Dataset, DatasetDict

ds = Dataset.from_pandas(df)
print('6' * 100)
print(df)
model_nm = 'microsoft/deberta-v3-small'
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print('7' * 100)
# tokz = AutoTokenizer.from_pretrained(model_nm, cache_dir='./')
tokz = AutoTokenizer.from_pretrained(model_nm)
print(tokz)

def tok_func(x): return tokz(x["input"])