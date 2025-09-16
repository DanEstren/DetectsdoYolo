from ultralytics import YOLO
import cv2
import supervision as sv
import time
import numpy as np

# Caminho do vídeo e modelo
video = "videos/ferrasm.mov"
model = YOLO(r"C:\Users\danil\Downloads\YoloProject\modelos\ToolsDetectorSegmentedNew\weights\best.pt").to(device="cuda")
out_path = "output/FerramentasVideo.mp4"

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(video)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
framecount = 0

# Configurações do tracker e anotadores
tracker = sv.ByteTrack()
RESIZE_SCALE = 0.6
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
polygon_annotator = sv.PolygonAnnotator()


# Configurações do vídeo de saída
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# Conjunto para rastrear IDs de vacas já contadas
counted_ids = set()
item_count = 0
# Callback que roda inferência em cada fatia
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

# Inicialize o slicer (exemplo com fatias 320x320 e 20% de sobreposição)
slicer = sv.InferenceSlicer(
    callback=callback,
    slice_wh=(320, 320),
    overlap_ratio_wh=(0.2, 0.2)
)

while cap.isOpened():
    start_time = time.time()  # Início para cálculo de FPS
    ret, frame = cap.read()
    if not ret:
        break
    framecount += 1

    # Redimensiona o frame
    frame = cv2.resize(frame, (width, height))
    detections = slicer(frame)
    # result = model(frame)[0]
    # detections = sv.Detections.from_ultralytics(result)
    # Segue processamento normal...
    detections = tracker.update_with_detections(detections)
    #classes=19 para vacas
    
    # Atualiza o tracker com as detecções
    detections = tracker.update_with_detections(detections)
    item_count = len(detections.tracker_id)

    annotated_frame = polygon_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Cria uma cópia para exibição com FPS e contagem
    cv2.putText(annotated_frame, f"Contados: {item_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    exibir = annotated_frame.copy()
    
    # Calcula e exibe FPS
    end_time = time.time()
    fps_display = round(1 / (end_time - start_time), 2)
    cv2.putText(exibir, f"FPS: {fps_display}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Exibe a contagem de vacas
       
    # Exibe o frame
    # cv2.imshow("video", exibir)
    
    # Salva o frame no vídeo de saída
    out.write(annotated_frame)
    
    print(f"Frame {framecount} de {total_frames}")
    
    # Sai se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
out.release()
cap.release()
cv2.destroyAllWindows()
