from speechbrain.pretrained import EncoderClassifier
import torchaudio
import torch
import os
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
import pandas as pd
import sqlite3


class SpeakerIdentifier:
    def __init__(self, dbname="voice_11"):
        self.dbname = dbname
        self.conn = sqlite3.connect(dbname)
        self.conn.enable_load_extension(True)
        self.conn.load_extension("./sqlite/vectorlite.so") # load vectorlite
        self.cursor = self.conn.cursor()
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")
        self.cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS user_feature using vectorlite (
            embedding float32[192] cosine,
            hnsw(max_elements=100)
        );
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS users ( 
            user_id integer primary key AUTOINCREMENT,
            name TEXT,
            sex TEXT,
            age INTEGER,
            height REAL,
            weight REAL
        );
        """)

    def get_user_embedding(self, signal):
        embedding = classifier.encode_batch(signal).squeeze().detach().cpu().numpy()
        return embedding

    def insert_user_information(self, name, sex="male", age=10, height=165, weight=70.0):
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE name = ?", (name,))
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO users(name, sex, age, height, weight) VALUES (?, ?, ?, ?, ?)",
                (name, sex, age, height, weight)
            )
            self.conn.commit()
            return True
        else:
            return False

    def select_user_id(self, name):
        cursor = self.conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE name = ?", (name,))
        result =  cursor.fetchone()
        return result[0]

    def insert_user_embedding(self, rowid, embedding):
        self.conn.execute("INSERT INTO user_feature(rowid, embedding) VALUES (?, ?)", (rowid, embedding))
        return self.conn.commit()

    def close(self):
        self.conn.close()

    def extract_clean_label(self, filename):
        # Extract speaker name (Alice, Brian, etc.)
        speaker_match = re.search(r'_(Alice|Brian|Charlotte|Jessica)_', filename)
        speaker = speaker_match.group(1) if speaker_match else "Unknown"

        return speaker   
        
if __name__ == "__main__":
    speaker_recognition = SpeakerIdentifier() 
    audio_dir = "./voice_sample/register_voice"
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3") or f.endswith(".wav")]
    audio_paths = {f: os.path.join(audio_dir, f) for f in audio_files}

    ## register
    for fname, path in audio_paths.items():
        signal, fs = torchaudio.load(path)
        speaker_name = speaker_recognition.extract_clean_label(fname)
        # print(speaker_name)
        speaker_recognition.insert_user_information(speaker_name)
        user_id = speaker_recognition.select_user_id(speaker_name)
        embedding = speaker_recognition.classifier.encode_batch(signal).squeeze().detach().cpu().numpy()
        # print(embedding)
        speaker_recognition.insert_user_embedding(user_id, embedding)

    audio_dir = "./voice_sample"
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3") or f.endswith(".wav")]
    audio_paths = {f: os.path.join(audio_dir, f) for f in audio_files}

    ## identification
    for fname, path in audio_paths.items():
        signal, fs = torchaudio.load(path)
        test_speaker_name = speaker_recognition.extract_clean_label(fname)
        embedding = speaker_recognition.classifier.encode_batch(signal).squeeze().detach().cpu().numpy()
        
        cursor = speaker_recognition.conn.cursor()
        result = cursor.execute(
                      "SELECT rowid FROM user_feature WHERE knn_search(embedding, knn_param(?, ?))",
                      (embedding, 1)
        ).fetchone()

        print(result)
        cursor.execute("SELECT name FROM users WHERE user_id = ?", (result[0],))
        result =  cursor.fetchone()
        print("test_speaker : %s, register_speaker : %s" % (test_speaker_name, result[0]))


    # audio_dir = "./voice_sample/register_voice"
    # audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3") or f.endswith(".wav")]
    # audio_paths = {f: os.path.join(audio_dir, f) for f in audio_files}

    # speaker_order = ['Jessica', 'Brian', 'Alice', 'Charlotte']

    # signal, fs = torchaudio.load(./voice_sample)

    # embeddings = {}
    # for fname, path in audio_paths.items():
    #     signal, fs = torchaudio.load(path)
    #     embedding = classifier.encode_batch(signal).squeeze().detach().cpu().numpy()
    #     embeddings[fname] = embedding

    # signal, fs = torchaudio.load(path)


