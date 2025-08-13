import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import json
import joblib
import itertools

class DataPreProcessor:
    def __init__(self, datatype, top_movie=10, test_size=0.2):
        self.datatype = datatype
        self.top_movie = top_movie
        self.test_size = test_size

    def preprocess(self):
        ## store user data
        users_columns=['user_id', 'age', 'sex', 'occupation', 'zipcode']
        users=[]

        with open('ml-100k/u.user','r',encoding='UTF-8') as f:
            for line in f.readlines():
                users.append(line.strip().split("|"))

        users_data = pd.DataFrame(np.array(users[0:]),columns=users_columns)
        users_data['sex'] = users_data['sex'].astype('category')
        users_data['sex_mapped'] = users_data['sex'].cat.codes
        sex_mapping = dict(enumerate(users_data['sex'].cat.categories))
        users_data['occupation'] = users_data['occupation'].astype('category')
        users_data['occupation_mapped'] = users_data['occupation'].cat.codes
        occupation_mapping = dict(enumerate(users_data['occupation'].cat.categories))
        users_data = users_data.drop(columns=['sex','occupation','zipcode'])
        users_data = users_data.rename(columns={'sex_mapped':'sex'})
        users_data = users_data.rename(columns={'occupation_mapped':'occupation'})

        train_user, test_user = train_test_split(users_data, test_size=self.test_size, random_state=42)

        ## store top-15 level movies' information(based on frequency)
        ratings_columns=['user_id', 'movie_id', 'rating', 'timestamp']
        ratings=[]
        with open('ml-100k/u.data','r',encoding='UTF-8') as f:
            for line in f.readlines():
                ratings.append(line.strip().split("\t"))
                
        rating_data=pd.DataFrame(np.array(ratings[0:]),columns=ratings_columns)
        top_movies=rating_data['movie_id'].value_counts().head(self.top_movie).index
        filtered_rating_data=rating_data[rating_data['movie_id'].isin(top_movies)].reset_index(drop=True)

        pivot_df = filtered_rating_data.pivot(index='user_id', columns=['movie_id'], values='rating')

        train_merged = pd.merge(train_user, pivot_df, on=['user_id'], how='left')
        train_merged = train_merged.fillna(0)
        test_merged = pd.merge(test_user, pivot_df, on=['user_id'], how='left')
        test_merged= test_merged.fillna(0)

        train_merged.to_csv("train_100k.csv", index=False)
        test_merged.to_csv("test_100k.csv", index=False)

if __name__ == "__main__":
    preprocess_100k = DataPreProcessor(datatype="100k") 
    preprocess_100k.preprocess()

   