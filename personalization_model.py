import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class AutoencoderModel:
    def __init__(self, input_dim, latent_dim=4):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.autoencoder = self._build_autoencoder()
        self.encoder = self._build_encoder()

    def _build_autoencoder(self):
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(24, activation='relu'),
            Dense(12, activation='relu'),
            Dense(8, activation='relu'),
            Dense(self.latent_dim, activation='relu'),  # bottleneck
            Dense(8, activation='relu'),
            Dense(12, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.input_dim, activation='sigmoid')
        ])
        return model

    def _build_encoder(self):
        # encoder는 autoencoder의 앞부분 레이어 4개만 복사
        encoder = Sequential(self.autoencoder.layers[:4])
        return encoder

    def compile(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        self.autoencoder.compile(optimizer=optimizer,
                                 loss=tf.keras.losses.MeanSquaredError())

    def fit(self, X, epochs=100, batch_size=32):
        history = self.autoencoder.fit(
            X, X, epochs=epochs, batch_size=batch_size, shuffle=True
        )
        return history

    def encode(self, X):
        return self.encoder.predict(X)

    def save(self, encoder_path, autoencoder_path):
        self.encoder.save(encoder_path)
        self.autoencoder.save(autoencoder_path)


class Trainer:
    def __init__(self, datafile, scaler_loc, encoder_loc, autoencoder_loc, converted_loc):
        self.datafile = datafile
        self.scaler_loc = scaler_loc
        self.encoder_loc = encoder_loc
        self.autoencoder_loc = autoencoder_loc
        self.converted_loc = converted_loc
        self.scaler = MinMaxScaler()

        if not os.path.exists(self.scaler_loc):
            os.makedirs(self.scaler_loc)
            print(f"folder create: {self.scaler_loc}")
        else:
            print(f"already exist: {self.scaler_loc}")

    def train(self):
        # 데이터 로드
        df = pd.read_csv(self.datafile)
        print(df)

        # 스케일링
        scaled_data = self.scaler.fit_transform(df)
        trans_df = pd.DataFrame(scaled_data, columns=df.columns)

        # 모델 초기화
        input_dim = trans_df.shape[1]
        model = AutoencoderModel(input_dim, latent_dim=4)
        model.compile(learning_rate=0.001)

        # 학습
        model.fit(trans_df, epochs=100, batch_size=32)

        X_encoded = model.encode(trans_df)
        print("Encoded shape: ", X_encoded.shape)
        print("Encoded contents: ", X_encoded)

        # 저장
        model.save(self.encoder_loc + '/encoder_model.h5', self.autoencoder_loc + '/autoencoder_model.h5')
        joblib.dump(self.scaler, self.scaler_loc + '/scaler.joblib')
        print("Encoder save to {}".format(self.encoder_loc))
        print("AutoEncoder save to {}".format(self.autoencoder_loc))
        print("Scaler save to {}".format(self.scaler_loc))

    def convert(self):
        # tflite로 변환

        model = load_model(self.encoder_loc)
        model.export('./lite_model/saved_model')
        converter = tf.lite.TFLiteConverter.from_saved_model('./lite_model/saved_model')
        tflite_model = converter.convert()
        with open(self.converted_loc + '/model.tflite', 'wb') as f:
            f.write(tflite_model)

        interpreter = tf.lite.Interpreter(model_path=self.converted_loc + '/model.tflite')
        interpreter.allocate_tensors()
        tensor_details = interpreter.get_tensor_details()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("Input Details :", input_details)
        print("Output Details :", output_details)

class Validator:
    def __init__(self, datafile, scaler_loc, autoencoder_loc):
        self.datafile = datafile
        self.scaler_loc = scaler_loc
        self.autoencoder_loc = autoencoder_loc
        self.scaler = MinMaxScaler()

    def overall_similarity(self, original_df, reconstructed_df):
        # 수치형만 선택
        num_df1 = original_df.select_dtypes(include=[np.number])
        num_df2 = reconstructed_df.select_dtypes(include=[np.number])
        
        # 코사인 유사도 (행 단위 평균)
        sim = cosine_similarity(num_df1, num_df2)
        return np.mean(np.diag(sim))

    def validate(self):
        df = pd.read_csv(self.datafile)
        df_transformed = df.copy()
        print(df_transformed)
        with open(self.scaler_loc + '/scaler.joblib', 'rb') as f:
            scaler = joblib.load(f)
        scaled_data = scaler.transform(df_transformed)
        trans_df = pd.DataFrame(scaled_data, columns=df_transformed.columns)
        print(trans_df)

        loaded_model = tf.keras.models.load_model(self.autoencoder_loc + "/autoencoder_model.h5")
        autoencoder_result = loaded_model.predict(trans_df)

        print(autoencoder_result)
        recover_data = scaler.inverse_transform(autoencoder_result)
        recover_df = pd.DataFrame(recover_data, columns=df.columns)

        similarity = self.overall_similarity(df, recover_df)
        print("similarity {}".format(similarity))


if __name__ == "__main__":
    trainer = Trainer(datafile="./train.csv", scaler_loc="./scaler", encoder_loc="./model", autoencoder_loc="./model", converted_loc="./tfmodel")
    trainer.train()

    validator = Validator(datafile="./test.csv", scaler_loc="./scaler", autoencoder_loc="./model")
    validator.validate()