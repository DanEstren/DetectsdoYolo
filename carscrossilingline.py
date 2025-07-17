from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np


video_path = "videos/vehiclesroad.mp4"
RESIZE_SCALE = 1
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
model = YOLO("yolo11l.pt")
tracker = sv.ByteTrack()
box_anottator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator(thickness=4)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_cont = 0
# print(sv.VideoInfo.from_video_path(video_path))
START = sv.Point(0, 400)
END = sv.Point(1280, 400)
line_zone = sv.LineZone(start=START, end=END)
line_zone_annotator = sv.LineZoneAnnotator(thickness=4,text_thickness=4,text_scale=2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_cont += 1
    frame = cv2.resize(frame,(width,height))
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)
    annottated_frame = box_anottator.annotate(frame,detections)
    labels = [
        f"{model.names[class_id] if hasattr(model, 'names') else f'Class_{class_id}'} (ID: {tracker_id})"
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)]  
    annottated_frame = label_annotator.annotate(annottated_frame,detections,labels=labels)
    annottated_frame = trace_annotator.annotate(annottated_frame,detections)
    line_zone.trigger(detections)
    annotated_frame = line_zone_annotator.annotate(annottated_frame, line_counter=line_zone)

    
    cv2.putText(annottated_frame, f"FRAME {frame_cont} // {total_frames}", (850, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("janela",annottated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()