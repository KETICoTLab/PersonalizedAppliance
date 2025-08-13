import paho.mqtt.client as mqtt
import vectorlite
import json

class DataCollector:
    def __init__(self, broker="192.0.0.1", port=1883, topics=None):
        self.broker = broker
        self.port = port
        self.topics = topics if topics else ["test/mqtt"]
        self.client = mqtt.Client()
        self.vectorlite = vectorlite.VectorLite()

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.topic_handlers = {
            "base/weather": self.handle_weather,
            "baser/bcg": self.handle_bcg,
            "base/control": self.handle_control,
            "base/voice": self.handle_voice
        }

    def on_connect(self, client, userdata, flags, rc):
       for topic in self.topics:
            client.subscribe(topic)
            print(f"Subscribed to topic: {topic}")

    def on_message(self, client, userdata, msg):
        if msg.topic in self.topic_handlers:
            self.topic_handlers[msg.topic](msg.payload.decode())
        else:
            print(f"Unhandled topic: {msg.topic}")

    def handle_weather(self, message):
        try:
            message_dict = json.loads(message)
            temperature = message_dict.get("temperature")
            humidity = message_dict.get("humidity")

            if temperature is not None and humidity is not None:
                self.vectorlite.insert_weather_info(temperature, humidity)
                print(f"[LOG] temperature: {temperature}, humidity: {humidity}")
            else:
                print("[ERROR] missing temperature or humidity in message.")

        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON format: {e}")

    def handle_bcg(self, message):
        try:
            message_dict = json.loads(message)
            bcg = message_dict.get("bcg")

            if bcg is not None:
                self.vectorlite.insert_user_bio_info(bcg)
                print(f"[LOG] bcg: {bcg}")
            else:
                print("[ERROR] missing bcg in message.")

        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON format: {e}")

    def handle_control(self, message):
        try:
            message_dict = json.loads(message)
            user_id = message_dict.get("user_id")
            selected_mode = message_dict.get("selected_mode")

            if user_id is not None and selected_mode is not None:
                self.vectorlite.insert_user_control_info(user_id, selected_mode)
                print(f"[LOG] user_id: {user_id}, selected_mode: {selected_mode}")
            else:
                print("[ERROR] Missing user_id or selected_mode in message.")

        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON format: {e}")

    # def handle_voice(self, message):
    #     try:
    #         message_dict = json.loads(message)
    #         voice = message_dict.get("voice")
    #         temperature = message_dict.get("temperature")
    #         temperature = message_dict.get("humidity")

    #         if voice is not None:
    #             # TODO: 

    #             self.vectorlite.insert_weather_info(temperature, humidity)
    #             print(f"[TEST] Temperature: {temperature}, Humidity: {humidity}")
    #         else:
    #             print("[ERROR] Missing voice in message.")

    #     except json.JSONDecodeError as e:
    #         print(f"[ERROR] Invalid JSON format: {e}")


    def start(self):
        """ MQTT 연결 시작 """
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_forever()

# 클래스 사용 예시
if __name__ == "__main__":
    mqtt_client = DataCollector()
    mqtt_client.start()