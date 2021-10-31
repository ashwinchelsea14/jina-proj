from jina import Flow, Executor, requests, Document, DocumentArray
import pandas as pd
import numpy as np

df = pd.read_csv('/home/ashwinkumar/Downloads/amaz.csv')
#df.head()

df = df.iloc[:,:5]
print(df.head())

df = df.drop_duplicates().dropna()
#df.shape

for i in range(5):
    print()
    print(f" product name : {df.iloc[i, 0]} ".center(80, ' '))
    print()
    print(df.iloc[i, -2])
    print()
    print('-' * 80)

docs = DocumentArray()
for i in range(df.shape[0]):
    name = df.iloc[i, 0]
    desc = df.iloc[i, -2]
    doc = Document(text = name)
    doc.tags['description'] = desc
    docs.append(doc)

#Flow

model = "sentence-transformers/paraphrase-distilroberta-base-v1"

flow = (
    Flow()
    .add(
        name="error_text_encoder",
        uses="jinahub://TransformerTorchEncoder",
        uses_with={"pretrained_model_name_or_path": model},
        install_requirements = True
    )
    .add(
        name="error_text_indexer",
        uses='jinahub://SimpleIndexer',
        install_requirements = True
    )
)

#Indexing
#!rm -rf workspace

with flow:
    flow.index(
      inputs=docs,
        docs = docs,
        parameters = {'name' : 'somethign', 'xyz' : 'asdf'}
  )

#Querying
query = Document(text = input('Query product : '))
with flow:
    response = flow.search(inputs = query, return_results = True)
