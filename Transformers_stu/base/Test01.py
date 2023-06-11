import os

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
creds = '{"username":"nanji890","key":"adb91d2252a05e279f331cf926046b4f"}'

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
if iskaggle:
    path = Path('../input/us-patent-phrase-to-phrase-matching')

import pandas as pd

df = pd.read_csv(path / 'train.csv')
print(df.columns)
print(df.shape)
print('0' * 100)
print(df.describe(include='object'))
print(df.head())
df['input'] = 'TEXT1: ' + df.context + '; TEXT2:' + df.target + '; ANC1: ' + df.anchor
print('1' * 100)
print(df.head())
print('2' * 100)
print(df.input.head())
from datasets import Dataset, DatasetDict

ds = Dataset.from_pandas(df)
print('3' * 100)
print(ds)
print('4' * 100)
print(df.columns)

model_nm = 'microsoft/deberta-v3-small'
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokz = AutoTokenizer.from_pretrained(model_nm)
print('5' * 100)
print(tokz)
print('6' * 100)
result = tokz.tokenize("G'day folks, I'm Jeremy from fast.ai!")
print(result)

print('7' * 100)
print(tokz.tokenize("G'day folks, I'm Jeremy from fast.ai!"))


def tok_func(x):
    return tokz(x["input"])


tok_ds = ds.map(tok_func, batched=True)
print('8' * 100)
print(tok_ds)
