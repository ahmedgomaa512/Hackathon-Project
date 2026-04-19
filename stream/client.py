import socket
import cv2
import pickle
import struct
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_frame

model = load_model("simple_model.h5")

classes = ["airplane","car","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('SERVER_IP', 9999))

data = b""
payload_size = struct.calcsize("Q")

while True:

    while len(data) < payload_size:
        data += client.recv(4096)

    packed = data[:payload_size]
    data = data[payload_size:]

    msg_size = struct.unpack("Q", packed)[0]

    while len(data) < msg_size:
        data += client.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data)

    img = preprocess_frame(frame)

    pred = model.predict(img)
    class_id = np.argmax(pred)
    confidence = np.max(pred)

    label = f"{classes[class_id]} ({confidence:.2f})"

    cv2.putText(frame, label, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    cv2.imshow("Live AI Stream", frame)

    if cv2.waitKey(1) == 27:
        break

client.close()
cv2.destroyAllWindows()