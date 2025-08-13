import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib


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
    def __init__(self, datatype):
        self.datatype = datatype
        self.scaler = MinMaxScaler()

    def train(self):
        # 데이터 로드
        df = pd.read_csv("train_32m.csv")
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

        # 인코딩
        X_encoded = model.encode(trans_df)
        print("Encoded shape:", X_encoded.shape)

        # 저장
        model.save('./encoder_model_32m.h5', './autoencoder_model_32m.h5')
        joblib.dump(self.scaler, "minmax_scaler_32m.joblib")
        print("Scaler saved to minmax_scaler_32m.joblib")


if __name__ == "__main__":
    trainer = Trainer(datatype="csv")
    trainer.train()