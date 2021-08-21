from keras.models import load_model
import cv2
import numpy as np



from pygame import mixer
mixer.init()
sound = mixer.Sound('alarm.wav')



model = load_model('../model/face-mask-detection.h5')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)


labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
  

while(True):

    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces=face_clsfr.detectMultiScale(gray)

    for (x,y,w,h) in faces:

        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(200,200))
        reshaped=np.reshape(resized,(1,200,200,3))
        result=model.predict(reshaped)

        print(f"Resultado: {result}")

        label=np.argmax(result,axis=1)[0]

        # cv2.rectangle(frame, (x, y), (x + w, y + h), color_dict[label],2)  # Bounding box (Big rectangle around the face)
        # cv2.rectangle(frame, (x, y - 40), (x + w, y), color_dict[label],-1)  # small rectangle above BBox where we will put our text
        # # Thickness of -1 px will fill the rectangle shape by the specified color.
        # cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),2)  # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/

        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],4)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],4)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_ITALIC, 1,(255,255,255),4)

        # if(labels_dict[label] =='MASK'):
        #    print("No Beep")
        # elif(labels_dict[label] =='NO MASK'):
        #         //sound.play()
        #         print("Beep")
        
    cv2.imshow('Mask Detection App',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



