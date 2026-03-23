from datasets import load_dataset
import os

"""
reference: https://medium.com/@patelneha1495/recommendation-system-in-python-using-als-algorithm-and-apache-spark-27aca08eaab3
"""


# 1. load review data and save as parquet 
parquet_name = 'my_amazon_books.parquet'
if not os.path.exists(parquet_name):
    print(f'The review dataset is being generated------')
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Books", trust_remote_code=True)
    dataset['full'].to_parquet('my_amazon_books.parquet')
else:
    print(f'{parquet_name} has been there----------')


# 2. load meta data and save as parquet
parquet_name = 'my_amazon_books_meta.parquet'
if not os.path.exists(parquet_name):
    print(f'The meta dataset is being generated------')
    meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Books", trust_remote_code = True)
    meta_dataset['full'].to_parquet('my_amazon_books_meta.parquet')
else:
    print(f'{parquet_name} has been there----------')