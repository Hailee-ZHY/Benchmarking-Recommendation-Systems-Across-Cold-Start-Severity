from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, hash, abs
from pyspark.sql.window import Window

import os

# 1. spark for review dataset
spark = SparkSession.builder\
    .appName('Amazon_book_recommendation')\
    .config('spark.driver.memory', '6g')\
    .config('spark.executor.memory', '6g')\
    .config("spark.default.parallelism", "32")\
    .config('spark.sql.shuffle.partitions', '32')\
    .config('spark.driver.maxResultSize', '1g')\
    .config("spark.dynamicAllocation.enabled", "false")\
    .config('spark.memory.fraction', '0.6')\
    .getOrCreate()

# 2. Preprocessing
class DataProcessing:
      def __init__(self, k):
            self.als_train_df, self.als_test_df = self._als_sample_data_(k)

      def _als_sample_data_(self, k):

            df_review = spark.read.parquet("my_amazon_books_sample.parquet").cache()

            # Step 1: filter active users: reviews > 6
            user_count = df_review.groupBy("user_id").agg(
                  F.count("*").alias("rating_cnt_by_user")
            )

            users_eligible = user_count.filter(
                  F.col("rating_cnt_by_user") >= 14
            ).select("user_id")

            df_filtered = df_review.join(users_eligible, on="user_id", how="inner")

            # Step 2: count again
            user_count = df_filtered.groupBy("user_id").agg(
                  F.count("*").alias("cnt")
            )

            df_filtered = df_filtered.join(user_count, on="user_id")

            # Step 3: assign random rank per user
            window = Window.partitionBy("user_id").orderBy(F.rand(42))  # fix the split

            df_ranked = df_filtered.withColumn(
                  "rank",
                  F.row_number().over(window)
            )

            # Step 4: split
            df_ranked = df_ranked.withColumn(
                  "threshold",
                  F.col("cnt") * k
            )

            test_df = df_ranked.filter(F.col("rank") <= F.col("threshold"))
            train_df = df_ranked.filter(F.col("rank") > F.col("threshold"))

            return train_df, test_df

      def als_data_overview(self):  # only for data review
            df_review = spark.read.parquet("my_amazon_books_sample.parquet").cache()

            # Step 1: filter active users: reviews > 6
            user_count = df_review.groupBy("user_id").agg(
                  F.count("*").alias("rating_cnt_by_user")
            )

            stats = user_count.agg(
                  F.max("rating_cnt_by_user").alias("user_review_max"),
                  F.min("rating_cnt_by_user").alias("user_review_min"),
                  F.avg("rating_cnt_by_user").alias("user_review_avg"),
            ).collect()[0]

            
            print(f"user_review_max: {stats['user_review_max']}")
            print(f"user_review_min: {stats['user_review_min']}")
            print(f"user_review_avg: {stats['user_review_avg']}")
            print(f"sample size: {df_review.count()}")
            print(f"user numbers: {df_review.select('user_id').distinct().count()}")
            print(f"item numbers: {df_review.select('parent_asin').distinct().count()}")

            user_count.select(
                  F.expr("percentile(rating_cnt_by_user, array(0.25, 0.5, 0.75, 0.9, 0.99))")
            ).show(truncate=False)

if __name__ == "__main__":
      d = DataProcessing(k = 0.2)
      print(f"count of train: {d.als_train_df.count()}") # => 154,349
      print(f"count of test: {d.als_test_df.count()}") # => 36,675




"""
sample data:
user_review_max: 3888
user_review_min: 1
user_review_avg: 8.246481828609797
sample size: 294754
user numbers: 35743
item numbers: 189305
+--------------------------------------------------------------------+
|percentile(rating_cnt_by_user, array(0.25, 0.5, 0.75, 0.9, 0.99), 1)|
+--------------------------------------------------------------------+
|[1.0, 2.0, 6.0, 14.0, 84.0]                                         |
+--------------------------------------------------------------------+
"""