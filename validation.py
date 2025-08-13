import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import json
import joblib
import pickle


df = pd.read_csv("test.csv")
df_transformed = df.copy()
print(df_transformed)
with open('minmax_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
scaled_data = scaler.transform(df_transformed)
trans_df = pd.DataFrame(scaled_data, columns=df_transformed.columns)
print(trans_df)


loaded_model = tf.keras.models.load_model("autoencoder_model.h5")
autoencoder_result = loaded_model.predict(trans_df)

print(autoencoder_result)
print(autoencoder_result.shape)

original_data = scaler.inverse_transform(autoencoder_result)

print(original_data)
