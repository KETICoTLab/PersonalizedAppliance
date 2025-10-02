import sqlite3
import pandas as pd
import numpy as np
import numpy as np
import datetime
import os
import pickle
import tensorflow as tf
import joblib
import time

def get_top_5_movies(row):
    ratings = row[4:].values
    top_5_indexes = ratings.argsort()[-5:][::-1]
    top_5_movies = row.index[4:][top_5_indexes].tolist()
    return top_5_movies

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

class DbInitializer:
    def __init__(self, dbname="test.db"):
        self.dbname = dbname
        self.conn = sqlite3.connect(dbname)
        self.conn.enable_load_extension(True)
        self.conn.load_extension("./sqlite/vectorlite.so") # load vectorlite
        self.cursor = self.conn.cursor()

    def initialize(self):      
        self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_component_test (
                    rowid integer primary key AUTOINCREMENT,
                    user_id integer,
                    age integer,
                    sex integer,
                    occupation integer,
                    best_pick1 integer,
                    best_pick2 integer,
                    best_pick3 integer 
                );
                """)

        self.cursor.execute("""
                CREATE VIRTUAL TABLE feature_embedding_test using vectorlite (
                    embedding float32[4] cosine, 
                    hnsw(max_elements=3000),'index_file.bin'
                );
                """)

        # 학습용 데이터를 입력
        df = pd.read_csv("train.csv")
        df_test = df.copy()
        df['top_3_movies'] = df.apply(get_top_3_movies, axis=1)

        start = time.time()
        encoder_model = tf.keras.models.load_model("./model/encoder_model.h5")
        with open('./scaler/scaler.joblib', 'rb') as f:
            scaler = joblib.load(f)

        scaled_data = scaler.transform(df_test)
        encoder_result = encoder_model(scaled_data)

        for row in df.itertuples(index=True, name="DataRow"):
            self.conn.execute("INSERT INTO embedding_component_test (user_id, age, sex, occupation, best_pick1, best_pick2, best_pick3) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (row.user_id, row.age, row.sex, row.occupation, row.top_3_movies[0], row.top_3_movies[1], row.top_3_movies[2]))

        i=1
        for row in range(encoder_result.shape[0]):
            try:
                self.conn.execute("INSERT INTO feature_embedding_test (rowid, embedding) VALUES (?, ?)",
                            (i, encoder_result[i-1].numpy().tobytes()))
                i=i+1
                # result = self.cursor.execute(
                #         "SELECT rowid FROM feature_embedding_test WHERE rowid=1"
                # ).fetchall()
                # print(result)

            except sqlite3.Error as e:
                print(e)

        self.conn.commit()
        self.conn.close()
        end = time.time()
        print(f" initialize elapsed time : {end - start:.5f} sec")

class Validator:
    def __init__(self, dbname="test.db"):
        self.dbname = dbname
        self.conn = sqlite3.connect(dbname)
        self.conn.enable_load_extension(True)
        self.conn.load_extension("./sqlite/vectorlite.so") # load vectorlite
        self.cursor = self.conn.cursor()

    def calculate_accuracy(self, row):
        pred_set = {row["pred_1"], row["pred_2"], row["pred_3"]}
        true_set = {row["true_1"], row["true_2"], row["true_3"]} 
        intersection = pred_set.intersection(true_set) 
        accuracy = len(intersection) / 3  
        return accuracy

    def validate(self):
        ## 검증은 테스트 데이터
        df = pd.read_csv("test.csv")
        df_test = df.copy()
        df['top_3_movies'] = df.apply(get_top_3_movies, axis=1)

        df[["recommend_1", "recommend_2", "recommend_3"]] = pd.DataFrame(df["top_3_movies"].tolist(), index=df.index)
        df[["recommend_1", "recommend_2", "recommend_3"]] = df[["recommend_1", "recommend_2", "recommend_3"]].astype(int)

        encoder_model = tf.keras.models.load_model("./model/encoder_model.h5")
        with open('./scaler/scaler.joblib', 'rb') as f:
            scaler = joblib.load(f)

        start = time.time()
        recommendation_list = []
        for row in df_test.itertuples(index=True, name="DataRow"):
            row_df = pd.DataFrame([row[1:]], columns=scaler.feature_names_in_)
            scaled_data = scaler.transform(row_df)
            vector = encoder_model(scaled_data)
            # print(vector[0].numpy())
            # scan_result = self.cursor.execute(
            #         "SELECT rowid FROM embedding_component_test where rowid=1"
            # ).fetchall()
            # print(scan_result)
            try:
                result = self.cursor.execute(
                        "SELECT rowid, distance FROM feature_embedding_test WHERE knn_search(embedding, knn_param(?, ?))", (vector[0].numpy().tobytes(), 1)
                ).fetchall()
            except sqlite3.Error as e:
                print(e)

            # print(result)

            placeholders = ", ".join("?" * len(result))
            query= f"SELECT best_pick1, best_pick2, best_pick3 FROM embedding_component_test WHERE rowid IN ({placeholders})"
            recommend_items = self.cursor.execute(query, (tuple(r[0] for r in result))).fetchall()
            recommendation_list.append([row.user_id] + list(recommend_items[0]))
        self.conn.commit()
        self.conn.close()

        end = time.time()
        print(f" validation_test elapsed time : {end - start:.5f} sec")

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

    tmp_conn = sqlite3.connect("test.db")
    tmp_cursor = tmp_conn.cursor()

    # 테이블 존재 여부 확인
    tmp_cursor.execute("""
        SELECT name 
        FROM sqlite_master 
        WHERE type='table' AND name='embedding_component_test';
    """)
    table_exists = tmp_cursor.fetchone() is not None\

    if not table_exists:
        initializer = DbInitializer() 
        initializer.initialize()

    validator = Validator() 
    validator.validate()

