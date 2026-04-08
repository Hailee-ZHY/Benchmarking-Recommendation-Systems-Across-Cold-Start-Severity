# ALS for CF methods

import math

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
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

    def ranking_eval(self, als_model, pipeline_model, df_test, top_k=10, relevance_threshold=4):
        # build relevant test interactions
        indexed_test = pipeline_model.transform(df_test).select(
            "user_id", "parent_asin", "user_index", "item_index", "rating"
        )
        relevant_test = indexed_test.filter(col("rating") >= relevance_threshold)

        # only evaluate users who have at least one relevant item in the test set
        eval_users = relevant_test.select("user_id", "user_index").distinct()
        if eval_users.count() == 0:
            return 0.0, 0.0

        # top-k recommendations for the evaluation users
        recommendations = als_model.recommendForUserSubset(
            eval_users.select("user_index"), top_k
        )

        rec_items = (
            recommendations
            .join(eval_users, on="user_index", how="inner")
            .select("user_id", F.posexplode("recommendations").alias("rank_idx", "rec"))
            .select(
                "user_id",
                (col("rank_idx") + F.lit(1)).alias("rank"),
                col("rec.item_index").alias("item_index"),
            )
        )

        actual_items = relevant_test.groupBy("user_id").agg(
            F.collect_set("item_index").alias("actual_items")
        )

        predicted_items = rec_items.groupBy("user_id").agg(
            F.expr(
                "transform("
                "sort_array(collect_list(named_struct('rank', rank, 'item_index', item_index))), "
                "x -> x.item_index"
                ")"
            ).alias("predicted_items")
        )

        ranking_input = (
            actual_items
            .join(predicted_items, on="user_id", how="inner")
            .select("predicted_items", "actual_items")
        )

        ranking_rows = ranking_input.collect()
        if not ranking_rows:
            return 0.0, 0.0

        recall_scores = []
        ndcg_scores = []

        for row in ranking_rows:
            predicted_items = row["predicted_items"] or []
            actual_items = set(row["actual_items"] or [])

            if not actual_items:
                continue

            hits = sum(1 for item in predicted_items[:top_k] if item in actual_items)
            recall_scores.append(hits / len(actual_items))

            dcg = 0.0
            for index, item in enumerate(predicted_items[:top_k], start=1):
                if item in actual_items:
                    dcg += 1.0 / math.log2(index + 1)

            ideal_hits = min(len(actual_items), top_k)
            idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        if not recall_scores:
            return 0.0, 0.0

        recall_at_k = sum(recall_scores) / len(recall_scores)
        ndcg_at_k = sum(ndcg_scores) / len(ndcg_scores)

        return recall_at_k, ndcg_at_k

    def run(self, k, top_k=10, relevance_threshold=4):
        d = DataProcessing(k)

        als_train_df = d.als_train_df
        als_test_df = d.als_test_df

        als_model, pipeline_model = self.train_asl(als_train_df)
        predictions = self.test_asl(als_model, pipeline_model, als_test_df)
        rmse, mae, coverage = self.asl_eval(predictions, als_test_df)
        recall_at_k, ndcg_at_k = self.ranking_eval(
            als_model,
            pipeline_model,
            als_test_df,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
        )

        # print(f"rmse: {rmse: .4f}")
        # print(f"mae: {mae: .4f}")
        # print(f"coverage: {coverage: .2%}")
        # print(f"recall@{top_k}: {recall_at_k: .4f}")
        # print(f"ndcg@{top_k}: {ndcg_at_k: .4f}")

        return rmse, mae, coverage, recall_at_k, ndcg_at_k

if __name__ == "__main__":
    a = als()
    rmse_holder = []
    mae_holder = []
    recall_holder = []
    ndcg_holder = []
    for k in [i/10 for i in range(1, 10)]:
        rmse, mae, _, recall_at_k, ndcg_at_k = a.run(k)
        rmse_holder.append(rmse)
        mae_holder.append(mae)
        recall_holder.append(recall_at_k)
        ndcg_holder.append(ndcg_at_k)
    print(f'rmse: {rmse_holder}')
    print(f'mae: {mae_holder}')
    print(f'recall@10: {recall_holder}')
    print(f'ndcg@10: {ndcg_holder}')
        
