from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, trim, isnan

import os

# 1. spark for review dataset
spark = SparkSession.builder\
    .appName('Amazon_book_recommendation')\
    .config('spark.driver.memory', '8g')\
    .config('spark.executor.memory', '8g')\
    .config("spark.default.parallelism", "64")\
    .config('spark.sql.shuffle.partitions', '64')\
    .config('spark.driver.maxResultSize', '2g')\
    .config("spark.dynamicAllocation.enabled", "false")\
    .getOrCreate()

# 2. Preprocessing
class DataProcessing:
        def __init__(self):
              spark.catalog.clearCache()
              # Review Data
              self.df_review = spark.read.parquet('my_amazon_books.parquet')
              # Meta Data
              self.df_meta = spark.read.parquet('my_amazon_books_meta.parquet')
              self.train_als, self.test_als, self.meta_als = self.Data_Process(method = 'ALS', size = 'small')
              self.cold_start_data = self.simulate(alpha = 10)
              

        def Data_Process(self, method = 'ALS', size = 'full'):
               if method == 'ALS' and size == 'full':
                      # only need: user_id, item_id, rating
                      df_review_clean = self.df_review.filter((col("user_id").isNotNull()) 
                                                              & (col("parent_asin").isNotNull())
                                                              & (trim(col("user_id")) != "")
                                                              & (trim(col("parent_asin")) != ""))

                      data_review_als = df_review_clean.select('user_id', 'parent_asin', 'rating') # this is used for training                     
                      data_meta_als = self.df_meta.select('title', 'parent_asin') # this is use for provide book title in the recommendation system
                      # split data into training and testing set
                      train_df, test_df = data_review_als.randomSplit(weights = [0.7, 0.3], seed = 42)
               elif method == 'ALS' and size == 'small':
                      df_review_clean = self.df_review.filter((col("user_id").isNotNull()) 
                                                              & (col("parent_asin").isNotNull())
                                                              & (trim(col("user_id")) != "")
                                                              & (trim(col("parent_asin")) != ""))

                      data_review_als = df_review_clean.select('user_id', 'parent_asin', 'rating').limit(100000) # this is used for training                     
                      data_meta_als = self.df_meta.select('title', 'parent_asin') # this is use for provide book title in the recommendation system
                      # split data into training and testing set
                      train_df, test_df = data_review_als.randomSplit(weights = [0.7, 0.3], seed = 42)
               
               return train_df, test_df, data_meta_als

### Simulate the sparse dataset (cold start)
        def simulate(self, alpha = 100, k = 1):
               # methods means the filter methods, we can choose ["both", "user", "item"]
               # k means the ratio of cold start

               # Step1. select users / items could be used for simulation

                user_rating_count = self.test_als.groupBy("user_id").agg(
                       F.count("rating").alias("rating_cnt_by_user")
                        )
                filtered_user_id = user_rating_count.filter(F.col("rating_cnt_by_user") >= alpha).select("user_id")

                item_rating_count = self.test_als.groupBy("parent_asin").agg(
                      F.count("rating").alias("rating_cnt_by_item")
                        )

                filtered_item_id = item_rating_count.filter(F.col("rating_cnt_by_item") >= alpha).select("parent_asin")
                     
                cold_simulate_df = self.test_als.join(filtered_user_id, on = "user_id", how = "inner") \
                                                .join(filtered_item_id, on = "parent_asin", how = "inner")
                """
                test_data: [user_id, parent_asin, rating]
                cold_simulate:[user_id, parent_asin, rating]
                """

               # step2. using K to remove rating record for users (temperarily only for users)
                # Todo
                return cold_simulate_df
                
      
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
        spark.catalog.clearCache()
        dp = DataProcessing()
        dp.cold_start_data.show(10)
        
        # Review Dataset
        # data_overview(df_review)
        # Meta Dataset
        # train, test, meta = dp.Data_Process()
        # print(f"train count: {dp.train_als.count()}")
        # print(f"test count: {dp.test_als.count()}")
        # print(f"meta count: {dp.meta_als.count()}")
        # user, item = dp.simulate()
        # print("user simulate data cols:")
        # for col, dtype in user.dtypes:
        #        print(f"{col}:{dtype}")

        # print("item simulate data cols:")
        # for col, dtype in item.dtypes:
        #        print(f"{col}:{dtype}")

        ## check again: does user_id has null value? -- confirm: there is no null value in user_id 
        # user_null_cnt = user.filter(col("user_id").isNull()).count()
        # print(user_null_cnt)

        # cold_df = dp.simulate()
        # cold_df.describe().show()

"""
Notes:
Data Summary for ALS: 
train count: 20,628,076                                                           
test count: 8,847,377                                                             
meta count: 4,448,181

Count the number of ratings for users and items:


1. user_id has non-value -- remove null value  -- still have null value
2. item shid has non-value -- remove null value


In test data (30%), we need to filter out users satifying the criteria. 
actually, the # of user parcitipainting into the test is much lower than 20% of the data set
Only 9639 rows. 
+-------+--------------------+--------------------+------------------+          
|summary|         parent_asin|             user_id|            rating|
+-------+--------------------+--------------------+------------------+
|  count|                9639|                9639|              9639|
|   mean|                    |                NULL| 4.201888162672477|
| stddev|                    |                NULL|0.9609044575690069|
|    min|                    |                    |               1.0|
|    max|                    |                    |               5.0|
+-------+--------------------+--------------------+------------------+
"""