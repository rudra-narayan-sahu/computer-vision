import cv2
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade=cv2.CascadeClassifier("haarcascade_smile.xml")
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        print("Could not read..")
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,5) 
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
            if len(eyes)>=2:
                cv2.putText(frame,"Eyes Detected",(x,y+h+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        smile=smile_cascade.detectMultiScale(roi_gray,1.7,20)
        for (sx,sy,sw,sh) in smile:
            if len(eyes)>=2:
                cv2.putText(frame,"Smiling",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        print("Quitting.......")
        break
cap.release()
cv2.destroyAllWindows()