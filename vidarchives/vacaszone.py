from ultralytics import YOLO
import cv2
import supervision as sv
import time as tm
import numpy as np
from datetime import datetime
import json
import uuid
# Caminho do vídeo e modelo
video = "videos/vacas.mp4"
model = YOLO(r"C:\Users\danil\Downloads\Atualizado\modelos\ModeloLargeMooVisionYolo11\weights\best.pt").to(device="cuda")
out_path = "outputs/vacaszonefinal.mp4"
output_file = "vacas_detectadas.json"
# Inicializa a captura de vídeo
cap = cv2.VideoCapture(video)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
framecount = 0

# Configurações do tracker e anotadores
tracker = sv.ByteTrack()
RESIZE_SCALE = 0.7
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Configurações do vídeo de saída
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# Definir a área de contagem (polígono)
counting_area = np.array([(108, 452), (155, 448), (192, 819), (136, 822)])
#segunda_counting = np.array([(46, 464),(53, 840),(92, 841),(77, 463)])

# Conjunto para rastrear IDs de vacas já contadas
counted_cow_ids = set()
cow_count = 0
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
    start_time = tm.time()  # Início para cálculo de FPS
    ret, frame = cap.read()
    if not ret:
        break
    framecount += 1

    # Redimensiona o frame
    frame = cv2.resize(frame, (width, height))

    # Realiza a detecção com YOLO
    result = model(frame, conf=0.6)[0]  #classes=19 para vacas
    detections = sv.Detections.from_ultralytics(result)
    
    # Atualiza o tracker com as detecções
    detections = tracker.update_with_detections(detections)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Verifica se o centro das caixas de detecção está dentro do polígono
    for xyxy, track_id, confidence in zip(detections.xyxy, detections.tracker_id,detections.confidence):
        # Calcula o centro da bounding box
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = (xyxy[1] + xyxy[3]) / 2
        center_point = (int(x_center), int(y_center))

        cv2.circle(frame, center=center_point, radius=5, color=(0, 0, 255), thickness=-1)
        cv2.putText(frame,f"ID :{track_id}",center_point,cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if cv2.pointPolygonTest(counting_area, center_point, False) > 0:
            #if cv2.pointPolygonTest(segunda_counting, center_point, False) > 0:
            if track_id not in counted_cow_ids:
                counted_cow_ids.add(track_id)
                cow_count += 1
                detection_data = {
                    "timestamp": current_time,
                    "tracker_id": int(track_id),
                    "class_name": "vaca",
                    "confidence": float(confidence)
                }
                # try:
                #     with open(output_file, 'r') as f:
                #         output_json = json.load(f)
                # except (json.JSONDecodeError, FileNotFoundError):
                #     output_json = []
                
                output_json.append(detection_data)

                # with open(output_file, 'w') as f:
                #     json.dump(output_json, f, indent=4)

    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.polylines(annotated_frame, pts=[counting_area], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.putText(annotated_frame, f"Vacas contadas: {cow_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Cria uma cópia para exibição com FPS e contagem
    exibir = annotated_frame.copy()
    
    # Calcula e exibe FPS
    end_time = tm.time()
    fps_display = round(1 / (end_time - start_time), 2)
    cv2.putText(exibir, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Exibe a contagem de vacas
    
    # Desenha o polígono

    #cv2.polylines(exibir, pts=[segunda_counting], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Exibe o frame
    cv2.imshow("video", exibir)
    
    # Salva o frame no vídeo de saída
    #out.write(annotated_frame)
    
    print(f"Frame {framecount} de {total_frames}")
    
    # Sai se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
out.release()
cap.release()
cv2.destroyAllWindows()
# Save detection data to JSON file
with open(output_file, 'w') as f:
    json.dump(output_json, f, indent=4)

# Imprime o total de vacas contadas ao final
print(f"Total de vacas contadas: {cow_count}")
print(f"Saved video to : {out_path}")