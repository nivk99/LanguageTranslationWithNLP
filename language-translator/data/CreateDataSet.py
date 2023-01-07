# data set
import pandas as pd

df = pd.read_csv("eng_-french.csv", encoding="utf-8")
# Separating English and French languages.
eng = df['English words/sentences']
fra = df['French words/sentences']

# en

with open("en", 'w', encoding='utf-8') as f:
    for i, v in enumerate(eng):
        f.write(v + '\n')
        if i ==1000:
            break

with open("small_vocab_en", 'r', encoding='utf-8') as f:
    data = f.read().split('\n')
    with open("en", 'a', encoding='utf-8') as f1:
        for i, v in enumerate(data):
            f1.write(v + '\n')

# fr

with open("fr", 'w', encoding='utf-8') as f:
    for i, v in enumerate(fra):
        f.write(v + '\n')
        if i==1000:
            break

with open("small_vocab_fr", 'r', encoding='utf-8') as f:
    data = f.read().split('\n')
    with open("fr", 'a', encoding='utf-8') as f1:
        for i, v in enumerate(data):
            f1.write(v + '\n')

