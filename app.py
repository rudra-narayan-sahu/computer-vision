import cv2
face_cas=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        print("Could not read..")
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cas.detectMultiScale(gray,1.1,5) 
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        print("Quitting.......")
        break
cap.release()
cv2.destroyAllWindows()