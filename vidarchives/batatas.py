from ultralytics import YOLO
import cv2
import supervision as sv
import time
import numpy as np

# Caminho do vídeo e modelo
video = "videos/potatos.mp4"
model = YOLO(r"C:\Users\danil\Downloads\Atualizado\modelos\PotatoConveyor50ModelM\weights\best.pt").to(device="cuda")
out_path = "outputs/batatasconveyorlen.mp4"

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

# Conjunto para rastrear IDs de vacas já contadas
counted_ids = set()
item_count = 0

while cap.isOpened():
    start_time = time.time()  # Início para cálculo de FPS
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
    item_count = len(detections.tracker_id)

    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Cria uma cópia para exibição com FPS e contagem
    cv2.putText(annotated_frame, f"Batatas contadas: {item_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    exibir = annotated_frame.copy()
    
    # Calcula e exibe FPS
    end_time = time.time()
    fps_display = round(1 / (end_time - start_time), 2)
    cv2.putText(exibir, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Exibe a contagem de vacas
       
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

