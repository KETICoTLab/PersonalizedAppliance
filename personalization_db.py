import sqlite3
import pandas as pd
import numpy as np
import numpy as np
import datetime
import vectorlite_py
import os

# vectorlite_path = os.environ.get("VECTORLITE_PATH", vectorlite_py.vectorlite_path())
#if vectorlite_path != vectorlite_py.vectorlite_path():
#    print(f"Using local vectorlite: {vectorlite_path}")
#vectolite_path = ".vectorlite.dll"


class VectorLite:
    def __init__(self, dbname):
        self.dbname = dbname
        self.conn = sqlite3.connect(dbname)
        self.conn.enable_load_extension(True)
        self.conn.load_extension("./sqlite/vectorlite.so") # load vectorlite
        self.cursor = self.conn.cursor()

    def table_initalization(self):

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS device (
            device_id integer primary key AUTOINCREMENT,
            name TEXT not null,
            description TEXT,
            owner TEXT,
            installTime DATETIME,
            installLocation TEXT
        );  
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            id integer primary key AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            temperature REAL,
            humidity REAL
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
        
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_bio ( 
            user_bio_id integer primary key AUTOINCREMENT,
            user_id integer,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            bcg REAL
        );
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_control_history (
            control_id integer primary key AUTOINCREMENT,
            user_id integer,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            selected_mode TEXT
        ); 
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_history ( 
            recommend_id integer primary key AUTOINCREMENT,
            user_id integer,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            recommendation TEXT
        );
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS personalized_model (
            id integer primary key AUTOINCREMENT,
            name TEXT,
            description TEXT,
            owner TEXT,
            installTime DATETIME,
            filepath TEXT
        );
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_component (
            rowid integer primary key AUTOINCREMENT,
            user_id integer,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            bcg REAL,
            temperature REAL,
            humidity REAL
        );
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_embedding ( 
            rowid integer primary key,
            embedding blob
        );
        """)

        # self.cursor.execute("""
        # CRREATE VIRTUAL TABLE feature_embedding using vectorlite (
        #     embedding float32[3],
        #     hnsw(max_elements=100)
        # );
        # """)
        
        return self.conn.commit()  
 

    # retrieve device information
    def select_device_info(self):
        self.cursor.execute("SELECT * FROM device")
        return self.cursor.fetchall()  
 
    # retrieve weather information
    def select_weather_info(self):
        self.cursor.execute("SELECT * FROM weather")       
        return self.cursor.fetchall()      
    
    def select_latest_weather_info(self):
        self.cursor.execute("SELECT * FROM weather ORDER BY timestamp DESC LIMIT 1")      
        return self.cursor.fetchall()  
    
    # retrieve user information
    def select_users_info(self):
        self.cursor.execute("SELECT * FROM users")
        return self.cursor.fetchall()  
        
    def select_user_info(self, user_id):
        self.cursor.execute("SELECT * FROM users where user_id = ?", (user_id,))
        return self.cursor.fetchall()  
    
    # retrieve user bio information
    def select_user_bios_info(self):
        self.cursor.execute("SELECT * FROM user_bio")
        return self.cursor.fetchall()  
    
    def select_user_bio_info(self, user_id):
        self.cursor.execute("SELECT * FROM user_bio where user_id = ?", (user_id,))    
        return self.cursor.fetchall()  
    
    # retrieve control information
    def select_user_controls_info(self):
        self.cursor.execute("SELECT * FROM user_control_history")
        return self.cursor.fetchall()  
    
    def select_user_control_info(self, user_id):
        self.cursor.execute("SELECT * FROM user_control_history where user_id = ?", (user_id,))    
        return self.cursor.fetchall()  
    
    # retrieve recommendation information
    def select_recommendations_info(self):
        self.cursor.execute("SELECT * FROM recommendation_history")
        return self.cursor.fetchall()  
    
    def select_recommendation_info(self, user_id):
        self.cursor.execute("SELECT * FROM recommendation_history where user_id = ?", (user_id,))    
        return self.cursor.fetchall()  
    
    # retrieve recommendation information
    def select_models_info(self):
        self.cursor.execute("SELECT * FROM personalized_model")
        return self.cursor.fetchall()  
            
    def select_embedding_component_info(self):
        self.cursor.execute("SELECT * FROM embedding_component")
        return self.cursor.fetchall()  
    
    def select_feature_embedding_info(self, vector, topk):
        # self.cursor.execute(
        #    "SELECT rowid FROM feature_embedding WHERE knn_search(embedding, knn_param(?, ?))",
        #    (vector, topk)
        # )
        self.cursor.execute(
            "SELECT rowid FROM feature_embedding order by vector_distance(?, embedding, 'l2') asc limit ?", (vector, topk)
        )
        return self.cursor.fetchall()  
    
    # insert statement
    def insert_device_info(self, name, description, owner, installTime, installLocation):
        self.conn.execute("INSERT INTO device(name, description, owner, installTime, installLocation) VALUES (?, ?, ?, ?, ?)", (name, description, owner, installTime, installLocation))
        return self.conn.commit()

    def insert_weather_info(self, temperature, humidity):
        self.conn.execute("INSERT INTO weather(temperature, humidity) VALUES (?, ?)", (temperature, humidity))
        return self.conn.commit()

    def insert_user_info(self, name, sex, age, height, weight):
        self.conn.execute("INSERT INTO users(name, sex, age, height, weight) VALUES (?, ?, ?, ?, ?)", (name, sex, age, height, weight))
        return self.conn.commit()

    def insert_user_bio_info(self, user_id, bcg):
        self.conn.execute("INSERT INTO user_bio(user_id, bcg) VALUES (?, ?)", (user_id, bcg))
        return self.conn.commit()

    def insert_user_control_info(self, user_id, selected_mode):
        self.conn.execute("INSERT INTO user_control_history(user_id, selected_mode) VALUES (?, ?)", (user_id, selected_mode))
        return self.conn.commit()

    def insert_recommendation_info(self, user_id, recommendation):
        self.conn.execute("INSERT INTO recommendation_history(user_id, recommendation) VALUES (?, ?)", (user_id, recommendation))
        return self.conn.commit()

    def insert_models_info(self, name, description, owner, installTime, filePath):
        self.conn.execute("INSERT INTO personalized_model(name, description, owner, installTime, filePath) VALUES (?, ?, ?, ?, ?)", (name, description, owner, installTime, filePath))
        return self.conn.commit()

    def insert_embedding_component_info(self, user_id, bcg, temperature, humidity):
        self.conn.execute("INSERT INTO embedding_component(user_id, bcg, temperature, humidity) VALUES (?, ?, ?, ?)", (user_id, bcg, temperature, humidity))
        return self.conn.commit()

    def insert_feature_embedding_info(self, rowid, embedding):
        try:
            print("rowid : ", rowid, "embedding : " , embedding)
            self.conn.execute("INSERT INTO feature_embedding(rowid, embedding) VALUES (?, ?)", (rowid, embedding))
            self.conn.commit()
        except sqlite3.Error as er:
            print('SQLlite error : %s' % (''.join(er.args)))
        return 1

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    vectorlite = VectorLite(dbname="personal_recommendation.db") 
    vectorlite.table_initalization()

    vectorlite.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = vectorlite.cursor.fetchall()
    print("Existing tables:", tables)
    print(vectorlite.cursor.execute('select vectorlite_info()').fetchall())

    # data = np.float32(np.random.random((10, 3)))
    # print(data[0].tobytes)
    # print(type(data[0].tobytes))

    data1 = np.array([1, 2, 3], dtype=np.float32)
    data2 = np.array([4, 5, 5], dtype=np.float32)
    data3 = np.array([6, 8, 12], dtype=np.float32)

    device_name="device_test"
    description="test info"
    owner="keti"
    installTime=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    installLocation="keti"
    vectorlite.insert_device_info(device_name, description, owner, installTime, installLocation)
    print(vectorlite.select_device_info())

    temperature=36.5
    humidity=56
    vectorlite.insert_weather_info(temperature, humidity)

    print(vectorlite.select_weather_info())
    print(vectorlite.select_latest_weather_info())
     
    user_name="wonki"
    sex="male"
    age=14
    height=159
    weight=23.5
    vectorlite.insert_user_info(user_name, sex, age, height, weight)

    print(vectorlite.select_users_info())
    print(vectorlite.select_user_info(1))

    user_id=1
    bcg=32.1
    vectorlite.insert_user_bio_info(user_id, bcg)
    print(vectorlite.select_user_bios_info())
    print(vectorlite.select_user_bio_info(user_id))

    selected_mode="physics"
    vectorlite.insert_user_control_info(user_id, selected_mode)
    print(vectorlite.select_user_controls_info())
    print(vectorlite.select_user_control_info(user_id))

    recommendation="relax"
    vectorlite.insert_recommendation_info(user_id, recommendation)
    print(vectorlite.select_recommendations_info())
    print(vectorlite.select_recommendation_info(user_id))

    model_name="autoencoder"
    filePath="www/xxx/xxx"
    vectorlite.insert_models_info(model_name, description, owner, installTime, filePath)
    
    print(vectorlite.select_models_info())

    vectorlite.insert_embedding_component_info(user_id, bcg, temperature, humidity)
    print(vectorlite.select_embedding_component_info())

    print("vector insert test start")

    vectorlite.insert_feature_embedding_info(1, data1.tobytes())
    vectorlite.insert_feature_embedding_info(2, data2.tobytes())
    vectorlite.insert_feature_embedding_info(3, data3.tobytes())
    print("vector insert test end")
    
    print("vector select test start")
    print(vectorlite.select_feature_embedding_info(data1.tobytes(), 10))
    print("vector select test end")
