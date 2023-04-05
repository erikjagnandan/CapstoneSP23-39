from ultralytics import YOLO
import time
import math
import cv2 as cv
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt") 
results = model("./Yard 4 3-6-23/image36.png")
tracker = DeepSort(max_age=5)
count = 0
for res in results:
    bbs = []
    for i in range(len(res.boxes.cls)):
        if res.boxes.cls[i] == 0:
            bbs.append((res.boxes.xywh[i],res.boxes.conf[i],0))
    tracks = tracker.update_tracks(bbs,frame=res.orig_img)
    for track in tracks:
        print(track)
        if not track.is_confirmed():
            print("this didnt get confirmed")
        track_id = track.track_id
        ltrb = track.to_ltrb()
        print(track_id)
        print("current location")
        print(res.boxes.xyxy)
        print("predicted location")
        print(ltrb)
    count+=1
print("it is done")