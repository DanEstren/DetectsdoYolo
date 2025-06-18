from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from datetime import datetime
import json
import os
import time as tm  # Renomeando para evitar conflito

class FPSBasedTimer:
    def __init__(self, fps):
        self.fps = fps
        self.times = {}

    def tick(self, detections):
        current_time_in_zone = []
        for tracker_id in detections.tracker_id:
            if tracker_id not in self.times:
                self.times[tracker_id] = 0
            self.times[tracker_id] += 1
            current_time_in_zone.append(self.times[tracker_id])
        
        return [frames / self.fps for frames in current_time_in_zone]

class RealTimeTimer:
    def __init__(self, json_output_path, area_name):
        self.start_times = {}  # {tracker_id: start_time}
        self.durations = {}    # {tracker_id: duration_in_seconds}
        self.active_durations = {}  # Para armazenar tempos acumulados
        self.json_output_path = json_output_path
        self.area_name = area_name
        # Initialize JSON file if it doesn't exist
        if not os.path.exists(self.json_output_path):
            with open(self.json_output_path, 'w') as f:
                json.dump([], f)

    def update(self, tracker_ids):
        current_time = tm.time()
        active_ids = set(tracker_ids)
        
        # Inicia timer para novos IDs e atualiza durações para ativos
        for tracker_id in tracker_ids:
            if tracker_id not in self.start_times:
                self.start_times[tracker_id] = current_time
                self.active_durations[tracker_id] = 0.0
            else:
                # Atualiza a duração acumulada enquanto está na zona
                self.active_durations[tracker_id] = current_time - self.start_times[tracker_id]
        
        # Remove IDs que não estão mais ativos e salva no JSON
        for tracker_id in list(self.start_times.keys()):
            if tracker_id not in active_ids:
                if tracker_id in self.active_durations:
                    self.durations[tracker_id] = self.active_durations[tracker_id]
                    # Save to JSON
                    self._save_to_json(tracker_id, self.durations[tracker_id])
                del self.start_times[tracker_id]
                del self.active_durations[tracker_id]
        
        # Prepara durações para retorno
        current_durations = []
        for tracker_id in tracker_ids:
            if tracker_id in self.durations:
                # Se já saiu antes, soma tempo anterior com tempo atual
                current_durations.append(self.durations[tracker_id] + self.active_durations.get(tracker_id, 0))
            else:
                # Ainda na zona, usa apenas o tempo ativo
                current_durations.append(self.active_durations.get(tracker_id, 0))
        
        return current_durations

    def _save_to_json(self, tracker_id, duration):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if (duration > 1):
            entry = {
                "tracker_id": int(tracker_id),
                "area": self.area_name,
                "time_spent_seconds": round(duration, 2),
                "exit_timestamp": current_time
            }
            # Read existing data
            try:
                with open(self.json_output_path, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                data = []
            
            # Append new entry
            data.append(entry)
            
            # Write back to file
            with open(self.json_output_path, 'w') as f:
                json.dump(data, f, indent=4)

# Initialize YOLO model, video capture, and supervision components
model = YOLO(r"C:\Users\danil\Downloads\Atualizado\modelos\MercadoYoutubeZoneModelS2\weights\best.pt")
cap = cv2.VideoCapture("videos/caixamercadocurto.mp4")
out_path = "outputs/mercadocaixacontas.mp4"
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Configurações do vídeo de saída
RESIZE_SCALE = 0.7
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0

# Initialize timers for each counting area with JSON output
json_output_path = "tracker_data.json"
timers = [
    RealTimeTimer(json_output_path, "Area 1"),
    RealTimeTimer(json_output_path, "Area 2")
]


data = []
with open(json_output_path, 'w') as f:
            json.dump(data, f, indent=4)

counting_area = np.array([(727, 125), (760, 555), (1031, 539), (887, 124)])
counting_area2 = np.array([(1021, 115), (1342, 94), (1341, 480), (1264, 503)])
mesa1 = np.array([(620, 279), (591, 454), (494, 460), (472, 573), (755, 557), (730, 268)])
mesa2 = np.array([(1026, 261), (1085, 424), (1001, 431), (1031, 533), (1256, 503), (1120, 253)])

while cap.isOpened():
    start_time = tm.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (width, height))
    
    frame_count += 1
    
    # Perform detection
    result = model(frame, conf=0.6)[0]
    detections = sv.Detections.from_ultralytics(result)
    
    # Update detections with tracker to get unique IDs
    detections = tracker.update_with_detections(detections)
    
    # Process each counting area
    people_in_areas = [[], []]  # Lista para armazenar detecções em cada área
    circle1 = 0
    circle2 = 0
    mesa1dt = 0
    mesa2dt = 0
    
    for det_idx, (bbox, class_id, tracker_id, confidence) in enumerate(zip(
        detections.xyxy, detections.class_id, detections.tracker_id, detections.confidence)):
        
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        center_point = (int(x_center), int(y_center))

        # Verifica se está em alguma área de contagem e é uma pessoa (class_id == 4)
        if class_id == 4:
            in_area1 = cv2.pointPolygonTest(counting_area, center_point, False) > 0
            in_area2 = cv2.pointPolygonTest(counting_area2, center_point, False) > 0
            
            if in_area1:
                people_in_areas[0].append(det_idx)
                circle1 += 1
            elif in_area2:
                people_in_areas[1].append(det_idx)
                circle2 += 1

        # Contagem nas mesas (mantido do código original)
        if cv2.pointPolygonTest(mesa1, center_point, False) > 0:
            cv2.circle(frame, center=center_point, radius=5, color=(0, 0, 255), thickness=-1)
            mesa1dt += 1
        
        
        if cv2.pointPolygonTest(mesa2, center_point, False) > 0:
            cv2.circle(frame, center=center_point, radius=5, color=(255, 255, 0), thickness=-1)
            mesa2dt += 1
    
    # Prepara labels com os timers para cada área
    labels = [
        f"{model.names[class_id] if hasattr(model, 'names') else f'Class_{class_id}'} (ID: {tracker_id})"
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)]  
    
    for area_idx, area_det_indices in enumerate(people_in_areas):
        if area_det_indices:
            area_detections = detections[area_det_indices]
            durations = timers[area_idx].update(area_detections.tracker_id)
            
            for i, det_idx in enumerate(area_det_indices):
                duration = durations[i]
                labels[det_idx] = f"#{area_detections.tracker_id[i]} {int(duration // 60):02d}:{int(duration % 60):02d}"
    
    # Annotate frame
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    # Draw polygons
    cv2.polylines(annotated_frame, pts=[counting_area], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(annotated_frame, pts=[counting_area2], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(annotated_frame, pts=[mesa1], isClosed=True, color=(0, 0, 255), thickness=1)
    cv2.polylines(annotated_frame, pts=[mesa2], isClosed=True, color=(255, 0, 255), thickness=1)
    
    # Display counters
    total_people = circle1 + circle2
    counter_text = f"Quantidade de Pessoas: {total_people}"
    cv2.putText(annotated_frame, counter_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Compras na bancada 1: {mesa1dt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Compras na bancada 2: {mesa2dt}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    print(f"Frame Atual {frame_count} de // {total_frames}")
    #out.write(annotated_frame)
    exibir = annotated_frame.copy()
    end_time = tm.time()  # Usando tm em vez de time
    fps_display = round(1 / (end_time - start_time), 2)
    cv2.putText(exibir, f"FPS: {fps_display}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show annotated frame
    cv2.imshow("janela", exibir)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processamento concluído!")