# ALS for CF methods

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

from DataProcesser import DataProcessing

class als:
    def __init__(self):
        pass

    def train_asl(self, 
            df_train, 
            user_col="user_id", 
            item_col="parent_asin", 
            rating_col="rating", 
            max_iter=10, # increase for better performance
            reg_param=0.09, 
            rank=30, # increase for better performace
            cold_start_strategy="drop"):
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

    def test_asl(self, als_model, pipeline_model, df_test):
        # index test data
        df_text_indexed = pipeline_model.transform(df_test)
        predictions = als_model.transform(df_text_indexed)

        return predictions

    def asl_eval(self, predictions, df_test):
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

    def run(self, k):
        d = DataProcessing(k)

        als_train_df = d.als_train_df
        als_test_df = d.als_test_df

        als_model, pipeline_model = self.train_asl(als_train_df)
        predictions = self.test_asl(als_model, pipeline_model, als_test_df)
        rmse, mae, coverage = self.asl_eval(predictions, als_test_df)

        # print(f"rmse: {rmse: .4f}")
        # print(f"mae: {mae: .4f}")
        # print(f"coverage: {coverage: .2%}")

        return rmse, mae, coverage

if __name__ == "__main__":
    a = als()
    rmse_holder = []
    mae_holder = []
    for k in [i/10 for i in range(1, 10)]:
        rmse, mae, _ = a.run(k)
        rmse_holder.append(rmse)
        mae_holder.append(mae)
    print(f'rmse: {rmse_holder}')
    print(f'mae: {mae_holder}')
        