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


class DbInitializer:
    def __init__(self, dbname="test.db"):
        self.dbname = dbname
        self.conn = sqlite3.connect(dbname)
        self.conn.enable_load_extension(True)
        self.conn.load_extension("./sqlite/vectorlite.so") # load vectorlite
        self.cursor = self.conn.cursor()

    def initialize(self):      
        self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_component_32m (
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

        # self.cursor.execute("""
        #         CREATE TABLE IF NOT EXISTS feature_embedding_32m ( 
        #             rowid integer primary key,
        #             embedding blob
        #         );
        #         """)

        self.cursor.execute("""
                CREATE VIRTUAL TABLE feature_embedding_32m using vectorlite (
                    embedding float32[4],
                    hnsw(max_elements=150000), 'index_file.bin'
                );
                """)

        df = pd.read_csv("train_32m.csv")
        df_test = df.copy()
        df['top_3_movies'] = df.apply(get_top_3_movies, axis=1)
        # print(df)

        start = time.time()
        encoder_model = tf.keras.models.load_model("./encoder_model_32m.h5")
        with open('minmax_scaler_32m.joblib', 'rb') as f:
            scaler = joblib.load(f)

        scaled_data = scaler.transform(df_test)
        encoder_result = encoder_model(scaled_data)

        for row in df.itertuples(index=True, name="DataRow"):
            # print(row)
            self.conn.execute("INSERT INTO embedding_component_32m (user_id, age, sex, occupation, best_pick1, best_pick2, best_pick3) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (row.user_id, row.age, row.sex, row.occupation, row.top_3_movies[0], row.top_3_movies[1], row.top_3_movies[2]))

        i=1
        for row in range(encoder_result.shape[0]):
            self.conn.execute("INSERT INTO feature_embedding_32m (rowid, embedding) VALUES (?, ?)",
                        (i, encoder_result[i-1].numpy().tobytes()))
            i=i+1

        self.conn.commit()
        end = time.time()
        print(f" initialize_32m elapsed time : {end - start:.5f} sec")
  
if __name__ == "__main__":
    initialize = DbInitializer() 
    initializer.initialize()
