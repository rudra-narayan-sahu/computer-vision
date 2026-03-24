import cv2
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime

# Configuration
DATASET_PATH = "dataset"
ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"

# Load dataset images and encode faces
def get_encodings():
    if os.path.exists(ENCODINGS_FILE):
        print("Loading encodings from file...")
        with open(ENCODINGS_FILE, 'rb') as f:
            return pickle.load(f)
    
    print("Encoding images from dataset...")
    images = []
    classNames = []
    myList = os.listdir(DATASET_PATH)
    
    for cl in myList:
        curImg = cv2.imread(f"{DATASET_PATH}/{cl}")
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
        else:
            print(f"⚠ Could not read image: {cl}")

    encodeListKnown = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encodeListKnown.append(encodings[0])
        else:
            print("⚠ No face found in one dataset image")

    # Save encodings
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((encodeListKnown, classNames), f)
    
    return encodeListKnown, classNames

# Attendance function
def markAttendance(name):
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    # Create file if it doesn't exist
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            f.write("Name,Time,Date")

    with open(ATTENDANCE_FILE, "r+") as f:
        data = f.readlines()
        attendance_records = [line.strip().split(",") for line in data]
        
        # Check if already marked for today
        already_marked = any(rec[0] == name and rec[2] == date_str for rec in attendance_records if len(rec) >= 3)

        if not already_marked:
            f.seek(0, 2) # Go to end of file
            f.write(f"\n{name},{time_str},{date_str}")
            print(f"✅ Attendance marked for {name}")

# Initialize
encodeListKnown, classNames = get_encodings()
print(f"Encoding Complete. Loaded {len(classNames)} faces.")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not detected")
    exit()

print("Starting Webcam...")

while True:
    success, img = cap.read()
    if not success:
        print("Camera read failed")
        break

    # Process smaller image for speed
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        if len(faceDis) == 0:
            continue

        matchIndex = np.argmin(faceDis)
        
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        if matches[matchIndex] and faceDis[matchIndex] < 0.6:
            name = classNames[matchIndex].upper()
            color = (0, 255, 0)
            status_text = f"Welcome {name}"
            markAttendance(name)
        else:
            name = "UNKNOWN"
            color = (0, 0, 255)
            status_text = "Not Recognized"

        # Draw UI
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Feedback message on top
        cv2.putText(img, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Recognition Attendance System", img)

    if cv2.waitKey(0): # ESC key
        break

cap.release()
cv2.destroyAllWindows()
