from keras.models import load_model
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN


cap = cv2.VideoCapture(0)
model = load_model('../model/new-face-mask-detection.h5')
# model = load_model('../model/detector.h5')
size = (200, 200)
detector = MTCNN()

while True:
    ret, frame = cap.read()

    faces = detector.detect_faces(frame)

    for face in faces:
        x1, y1, w, h = face['box']
        x2 = x1 + w
        y2 = y1 + h

        roi = frame[y1: y2, x1:x2]

        if np.sum([roi]) != 0:

            roi = cv2.resize(roi, size)
            roi = (roi.astype('float')/255.0)

            roi = np.reshape(roi, [1, 200, 200, 3])

            pred = model.predict([[roi]])

            pred = pred[0]

            print(pred)

            if pred[0] >= pred[1]:
                label = 'NO MASK'
                color = (0,0,255)
            else:
                label = 'MASK'
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)



    cv2.imshow("MASK DETECTOR", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
