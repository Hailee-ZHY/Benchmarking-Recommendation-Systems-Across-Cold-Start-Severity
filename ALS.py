# ALS for CF methods

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

from DataProcesser import DataProcessing

processer = DataProcessing()

df_train = processer.train_als
df_test = processer.cold_start_data  # can be changed back to test_als

def train_asl(
    df_train, 
    user_col="user_id", 
    item_col="parent_asin", 
    rating_col="rating", 
    max_iter=3, #5
    reg_param=0.09, 
    rank=10, #25 
    cold_start_strategy="drop"
):
    # index columns
    user_indexer = StringIndexer(inputCol=user_col, outputCol="user_index", handleInvalid="skip")
    item_indexer = StringIndexer(inputCol=item_col, outputCol="item_index", handleInvalid="skip")

    pipeline = Pipeline(stages=[user_indexer, item_indexer])
    df_train_indexed = pipeline.fit(df_train).transform(df_train)
    
    # train als 
    als = ALS(
        maxIter=max_iter, 
        regParam=reg_param, 
        rank=rank,
        userCol="user_index", 
        itemCol="item_index", 
        ratingCol=rating_col, 
        coldStartStrategy=cold_start_strategy,
    )

    als_model = als.fit(df_train_indexed)
    return als_model, pipeline

def test_asl(als_model, pipeline, df_test, user_col, item_col):
    # index test data
    df_text_indexed = pipeline.transform(df_test)
    predictions = als_model.transform(df_text_indexed)

    return predictions

if __name__ == "__main__":
    als_model, pipeline = train_asl(df_train)
    predictions = test_asl(als_model, pipeline, df_test, "user_id", "parent_asin")

    predictions.show(10)