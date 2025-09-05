import sqlite3
import pandas as pd
import numpy as np
import numpy as np
import datetime
# import vectorlite_py
import os
import pickle
import tensorflow as tf
import joblib
import time
from tflite_runtime.interpreter import Interpreter

def get_top_3_movies(row):
    ratings = row[4:].values
    top_3_indexes = ratings.argsort()[-3:][::-1]
    top_3_movies = row.index[4:][top_3_indexes].tolist()
    return top_3_movies

def calculate_accuracy(row):
    pred_set = {row["pred_1"], row["pred_2"], row["pred_3"]}
    true_set = {row["true_1"], row["true_2"], row["true_3"]} 
    intersection = pred_set.intersection(true_set) 
    accuracy = len(intersection) / 3  
    return accuracy


class Validator:
    def __init__(self, dbname="test.db"):
        self.dbname = dbname
        self.conn = sqlite3.connect(dbname)
        self.conn.enable_load_extension(True)
        self.conn.load_extension("./sqlite/vectorlite.so") # load vectorlite
        self.cursor = self.conn.cursor()

    def validate(self):
        ## store user data
        df = pd.read_csv("test_100k.csv")
        df_test = df.copy()
        df['top_3_movies'] = df.apply(get_top_3_movies, axis=1)

        df[["recommend_1", "recommend_2", "recommend_3"]] = pd.DataFrame(df["top_3_movies"].tolist(), index=df.index)
        df[["recommend_1", "recommend_2", "recommend_3"]] = df[["recommend_1", "recommend_2", "recommend_3"]].astype(int)

        encoder_model = tf.keras.models.load_model("./encoder_model_100k.h5")
        with open('minmax_scaler_100k.joblib', 'rb') as f:
            scaler = joblib.load(f)

        start = time.time()
        recommendation_list = []
        for row in df_test.itertuples(index=True, name="DataRow"):
            row_df = pd.DataFrame([row[1:]], columns=scaler.feature_names_in_)
            scaled_data = scaler.transform(row_df)
            vector = encoder_model(scaled_data)
            result =self.cursor.execute(
                    "SELECT rowid FROM feature_embedding_100k order by vector_distance(?, embedding, 'cosine') asc limit ?", (vector, 1)
            ).fetchall()
            placeholders = ", ".join("?" * len(result))
            query= f"SELECT best_pick1, best_pick2, best_pick3 FROM embedding_component_100k WHERE rowid IN ({placeholders})"
            recommend_items = self.cursor.execute(query, (tuple(r[0] for r in result))).fetchall()
            recommendation_list.append([row.user_id] + list(recommend_items[0]))
        self.conn.commit()

        end = time.time()
        print(f" validation_100k elapsed time : {end - start:.5f} sec")

        recommendation_df = pd.DataFrame(recommendation_list, columns=["user_id", "recommend_1", "recommend_2", "recommend_3"])
        df_merged = recommendation_df.merge(df, on="user_id", suffixes=("_pred", "_true"))

        df_merged["accuracy"] = df_merged.apply(
            lambda row: calculate_accuracy({
                "pred_1": row["recommend_1_pred"], "pred_2": row["recommend_2_pred"], "pred_3": row["recommend_3_pred"],
                "true_1": row["recommend_1_true"], "true_2": row["recommend_2_true"], "true_3": row["recommend_3_true"]
            }), axis=1
        )
        average_accuracy = df_merged["accuracy"].mean()
        print(f"\n 100k 전체 평균 정확도: {average_accuracy:.4f}")

if __name__ == "__main__":
    validator = Validator() 
    validator.validate()



