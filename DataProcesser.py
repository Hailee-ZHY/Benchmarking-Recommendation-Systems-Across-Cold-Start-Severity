from pyspark.sql import SparkSession
import os

# 1. spark for review dataset
spark = SparkSession.builder\
    .appName('Amazon_book_recommendation')\
    .config('spark.driver.memory', '4g')\
    .config('spark.executor.memory', '2g')\
    .config('spark.sql.shuffle.partitions', '20')\
    .config('spark.memory.fraction', '0.6')\
    .config('spark.driver.maxResultSize', '1g')\
    .getOrCreate()

# 2. Preprocessing
class DataProcessing:
        def __init__(self):
              # Review Data
              self.df_review = spark.read.parquet('my_amazon_books.parquet')
              # Meta Data
              self.df_meta = spark.read.parquet('my_amazon_books_meta.parquet')

        def Data_Process(self, method = 'ALS'):
               if method == 'ALS':
                      # only need: user_id, item_id, rating
                      data_review_als = self.df_review.select('user_id', 'parent_asin', 'rating')
                      data_meta_als = self.df_meta.select('title', 'parent_asin')
               return data_review_als, data_meta_als

### Simulate the sparse dataset (cold start)
# TODO: k can be the arg passed to the function


# 2. Overview of dataset
def data_overview(df):
        print(f'review data summary: {df.count()}')
        print(f'review data columns: {df.columns}')
        print(f'data type: {df.dtypes}')

        ## for review dataset
        if {'title', 'text'}.issubset(df.columns):
            print(f'check title and text: ')
            df.select('title', 'text').show(5, truncate = True)

if __name__ == "__main__":
        dp = DataProcessing()
        # Review Dataset
        # data_overview(df_review)
        # Meta Dataset
        data_overview(dp.df_meta)