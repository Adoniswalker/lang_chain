import os
import pinecone
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
print(pinecone.list_indexes())
# print(pinecone.info.version())

# creating index
index_name = 'dennis-langchain-index'


def create_index():
    if index_name not in pinecone.list_indexes():
        print(f'Creating index {index_name} ...')
        pinecone.create_index(index_name, dimension=1536, metric='cosine', pods=1, pod_type='p1.x2')
        print('Done')
    else:
        print(f'Index {index_name} already exists')

def describe_index():
    print(pinecone.describe_index(index_name))

def delete_index():
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)

# inseting into pinecone index
def insert_pinecone():
    import random
    vectors = [[random.random() for _ in range(1536)] for v in range(5)]
    # vectors
    ids = list('abcde')
    index = pinecone.Index(index_name)
    index.upsert(vectors=zip(ids, vectors))


# insert_pinecone()
index = pinecone.Index(index_name)
print(index.describe_index_stats())
index.upsert([('a', [0.3]*1536)])
# index.delete(ids=['a'])
# print(index.fetch(ids=['a']))
import random
query = [[0.3 for _ in range(1536)] for v in range(1)]
result = index.query(
    queries=query,
    top_k=3,
    include_values=False
)
print(result)