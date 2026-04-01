# ALS for CF methods

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

from DataProcesser import DataProcessing

processer = DataProcessing()

df_train = processer.train_als.limit(2000) # only use the first 2000 for testing since the dataset is too big
# df_test = processer.cold_start_data.limit(600) # cold_start data for testing
df_test = processer.test_als.limit(600) # test data for testing

def train_asl(
    df_train, 
    user_col="user_id", 
    item_col="parent_asin", 
    rating_col="rating", 
    max_iter=1, #5
    reg_param=0.09, 
    rank=10, #25 
    cold_start_strategy="drop"
):
    # index columns
    user_indexer = StringIndexer(inputCol=user_col, outputCol="user_index", handleInvalid="skip")
    item_indexer = StringIndexer(inputCol=item_col, outputCol="item_index", handleInvalid="skip")

    pipeline = Pipeline(stages=[user_indexer, item_indexer])
    pipeline_model = pipeline.fit(df_train)
    df_train_indexed = pipeline_model.transform(df_train)
    
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
    return als_model, pipeline_model

def test_asl(als_model, pipeline_model, df_test, user_col, item_col):
    # index test data
    df_text_indexed = pipeline_model.transform(df_test)
    predictions = als_model.transform(df_text_indexed)

    return predictions

def asl_eval(predictions):
    # predictions is the result returned from test_asl function
    ## 1) filter out null value in the prediction results
    pred = predictions.filter(col("prediction").isNotNull())
    
    ## 2) Evaluation metrics: RMSE and MAE
    rmse = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    ).evaluate(pred)

    mae = RegressionEvaluator(
        metricName="mae", labelCol="rating", predictionCol="prediction"
    ).evaluate(pred)

    ## 3) coverage? How many samples get the prediction results
    coverage = pred.count() / df_test.count()

    return rmse, mae, coverage


if __name__ == "__main__":
    als_model, pipeline_model = train_asl(df_train)
    predictions = test_asl(als_model, pipeline_model, df_test, "user_id", "parent_asin")
    predictions.show(10)
    rmse, mae, coverage = asl_eval(predictions)
    print(f"rmse: {rmse: .4f}")
    print(f"mae: {mae: .4f}")
    print(f"coverage: {coverage: .2%}")
