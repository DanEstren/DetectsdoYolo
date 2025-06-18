from ultralytics import YOLO
import cv2
import supervision as sv
import time
import numpy as np
import json
from datetime import datetime
import uuid

video = "videos/MercadoNew.mp4"
model = YOLO(r"C:\Users\danil\Downloads\Atualizado\modelos\MercadoNewS2\weights\best.pt")
out_path = "outputs/MercadoNewVid.mp4"
cap = cv2.VideoCapture(video)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
framecount = 0
RESIZE_SCALE = 0.5
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
tracker = sv.ByteTrack()
counting_area = np.array([(475, 81),(364, 316), (563, 368),(642, 93)])
mercadoitens = set()
prod_scan = 0
tracker_frame_counts = {}
output_file= "outputs/mercadoNew.json"

first_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
unique_id = str(uuid.uuid4())
init_data = {
    "ID da Execucao do Programa" : unique_id,
    "Execucao do Programa as" : first_time,
}
try:
    with open(output_file, 'r') as f:
        output_json = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    output_json = []
                
output_json.append(init_data)

with open(output_file, 'w') as f:
     json.dump(output_json, f, indent=4)

while cap.isOpened():
    start_time = time.time()  # Start time for FPS calculation
    ret, frame = cap.read()
    if not ret:
        break
    framecount += 1
    frame = cv2.resize(frame, (width, height))
    result = model(frame,conf=0.6)[0] #classes=[0] no caso do model, ele começou a partir do 1
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections) #isso tmb remover se necessário


    for xyxy, track_id, confidence in zip(detections.xyxy, detections.tracker_id,detections.confidence):
        # Calcula o centro da bounding box
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = (xyxy[1] + xyxy[3]) / 2
        center_point = (int(x_center), int(y_center))

        cv2.circle(frame, center=center_point, radius=5, color=(0, 0, 255), thickness=-1)
        cv2.putText(frame,f"ID :{track_id}",center_point,cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if cv2.pointPolygonTest(counting_area, center_point, False) > 0:
            # Update frame counts for detected tracker IDs
            current_tracker_ids = set(detections.tracker_id)
            for tracker_id in current_tracker_ids:
                tracker_frame_counts[tracker_id] = tracker_frame_counts.get(tracker_id, 0) + 1

        labels = [
        f"{model.names[class_id] if hasattr(model, 'names') else f'Class_{class_id}'} (ID: {tracker_id})"
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)] 


        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if (cv2.pointPolygonTest(counting_area, center_point, False) > 0 and
            tracker_frame_counts.get(tracker_id, 0) > 2) :
            if track_id not in mercadoitens:
                mercadoitens.add(track_id)
                prod_scan +=1
                detection_data = {
                    "timestamp": current_time,
                    "tracker_id": int(track_id),
                    "class_name": labels,
                    "confidence": float(confidence)
                }
                try:
                    with open(output_file, 'r') as f:
                        output_json = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    output_json = []
                
                output_json.append(detection_data)

                with open(output_file, 'w') as f:
                    json.dump(output_json, f, indent=4)
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    
    # Create a copy for display with FPS
    cv2.polylines(annotated_frame, pts=[counting_area], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.putText(annotated_frame, f"Quantidade de Produtos: {prod_scan}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    exibir = annotated_frame.copy()
    # Calculate FPS
    end_time = time.time()
    fps_display = round(1 / (end_time - start_time), 2)

    cv2.putText(exibir, f"FPS: {fps_display}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(annotated_frame)
    print(f"Frame {framecount} de {total_frames}")
    # o Exibir, faz com que ele salve o vídeo, sem precisar mostrar que os FPS 
    cv2.imshow("video", exibir)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()