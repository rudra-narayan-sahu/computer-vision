import cv2
from ultralytics import YOLO
model=YOLO("yolo26x-obb.pt")
results = model("videoplayback (1).mp4", stream=True)

for r in results:
    frame = r.plot()
    cv2.imshow("YOLO Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()