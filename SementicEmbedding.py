from DataProcesser import DataProcessing
from sentence_transformers import SentenceTransformer
import pandas as pd 
import numpy as np 
from pyspark.sql import functions as F
import torch 
# create embedding space

class semnetic_embedding:
      def __init__(self,train_df, test_df, meta_data, top_n, k, model = "all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model, device = 'cuda')
            self.train_df = train_df
            self.test_df = test_df
            self.top_n = top_n
            self.k = k
            self.meta_data = meta_data

      def generate_item_embedding(self):
            # transoform to pandas 
            meta_data_df = self.meta_data.toPandas()
            # generating embedding space
            embedding = self.model.encode(
                  meta_data_df["embedding_text"].tolist(), 
                  show_progress_bar = True, 
            )

            meta_data_df["embedding"] = list(embedding)

            self.item_embedding_df = meta_data_df
            self.asin_to_embedding = dict(zip(meta_data_df["parent_asin"], meta_data_df["embedding"]))

            return meta_data_df
      
      def user_history(self):
            # transform it to pandas
            user_history_df = (
                  self.train_df.groupBy("user_id")\
                  .agg(F.collect_list("parent_asin").alias("parent_asin"))\
                  .toPandas()
            )
            
            user_profile = {}

            for _, row in user_history_df.iterrows():
                  user_id = row["user_id"]
                  item_list = row["parent_asin"]

                  item_vectors = [
                        self.asin_to_embedding[item] for item in item_list if item in self.asin_to_embedding
                  ]

                  if len(item_vectors) > 0:
                        user_profile[user_id] = np.mean(item_vectors, axis=0)

            self.user_profile = user_profile
            return user_profile
      
      def recommand_for_user(self, user_id):
            if user_id not in self.user_profile:
                  return []
            
            user_vector = self.user_profile[user_id]

            # avoid repeat recommend 
            seen_items = (
                  self.train_df.filter(F.col("user_id") == user_id)\
                              .select("parent_asin")\
                              .distinct()\
                              .toPandas()["parent_asin"]\
                              .tolist()
            )
            seen_items = set(seen_items)

            candidates = self.item_embedding_df[~self.item_embedding_df["parent_asin"].isin(seen_items)].copy()

            if candidates.empty:
                  return []
            
            item_matrix = np.vstack(candidates["embedding"].values)

            # cosine similarity
            user_norm = np.linalg.norm(user_vector)
            item_norms = np.linalg.norm(item_matrix, axis = 1)
            sims = np.dot(item_matrix, user_vector) / (item_norms * user_norm + 1e-12)

            candidates["score"] = sims
            recs = candidates.sort_values(by = "score", ascending = False).head(self.top_n)

            return recs["parent_asin"].tolist()
      
      def embedding_eval(self):
            test_truth_df = (
                  self.test_df.groupBy("user_id")\
                              .agg(F.collect_set("parent_asin").alias("parent_asin"))
                              .toPandas()
            )

            precision_list = []
            recall_list = []
            hit_list = []

            for _, row in test_truth_df.iterrows():
                  user_id = row["user_id"]
                  item_list = set(row["parent_asin"])

                  resc = self.recommand_for_user(user_id)
                  resc_set = set(resc)

                  if len(resc) == 0:
                        continue 
                        
                  hits = len(resc_set & item_list)

                  precision = hits / self.top_n
                  recall = hits / len(item_list) if len(item_list) > 0 else 0
                  hit_rate = 1 if hits > 0 else 0

                  precision_list.append(precision)
                  recall_list.append(recall)
                  hit_list.append(hit_rate)

                  result = {
                        "precision_at_k": float(np.mean(precision_list)) if precision_list else 0.0,
                        "recall_at_k": float(np.mean(recall_list)) if recall_list else 0.0,
                        "hit_rate_at_k": float(np.mean(hit_list)) if hit_list else 0.0,
                  }

            return result

      def run(self):
            self.generate_item_embedding()
            self.user_history()
            # self.recommand_for_user()
            result = self.embedding_eval()
            return result
            

if __name__ == "__main__":
      k = 0.5
      top_n = 10 
      d = DataProcessing(k)
      train_df, test_df, meta_data = d.als_train_df, d.als_test_df, d.embedding_meta_df
      s = semnetic_embedding(train_df, test_df, meta_data, top_n, k)
      eval_res = s.run()
      print(f"embedding result: {eval_res}")     

      
      # result = []
      # for k in [i/10 for i in range(1, 10)]:
      #       s = semnetic_embedding(train_df, test_df, meta_data, top_n, k)
      #       eval_res = s.run()
      #       result.append({
      #             "k": k, 
      #             "precision": eval_res["precision_at_k"],
      #             "recall": eval_res["recall_at_k"], 
      #             "hit_rate": eval_res["hit_rate_at_k"]
      #       })
      # result_df = pd.DataFrame(result)
      # result_df.to_csv("embedding_eval_by_k.csv", index = False)
