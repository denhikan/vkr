import torch
import tensorflow as tf
import cv2
import numpy as np
from ultralytics import YOLO

model_yolo = YOLO('yolov8.pt')

model_tf = tf.keras.models.load_model('my_model(tensorflow,keras).h5')

classes1 = {
    0: '(20km/h)',
    1: '(30km/h)',
    2: '(50km/h)',
    3: '(60km/h)',
    4: '(70km/h)',
    5: '(80km/h)',
    6: '(90km/h)',
    7: '(100km/h)',
    8: '(120km/h)',
}


cap = cv2.VideoCapture('C:/Code/ansambel/znaki/Прошли/VID_20230516_202428.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
new_width, new_height = 1280, 720

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 120, (new_width, new_height))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (new_width, new_height))
 
    results = model_yolo(frame, classes=[2])
    detections = results[0].boxes
    if len(detections) > 0:
        for *xyxy, conf, cls in detections.boxes:
            if conf > 0.9:
                if xyxy: 
                    x1, y1, x2, y2 = map(int, xyxy)
                else:
                    print('Координаты не найдены')
                    continue

                sign = frame[y1:y2, x1:x2]
                sign = cv2.resize(sign, (30, 30))
                cv2.imshow('sign',sign)
                cv2.waitKey(0)
                sign = np.expand_dims(sign, axis=0)
   
                sign_class = model_tf.predict(sign)
                sign_class = np.argmax(sign_class, axis=1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
                if sign_class[0] in classes1:
                    sign_text = classes1[sign_class[0]]
                    cv2.putText(frame, sign_text, (x1+45, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                print(sign_class[0])
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
