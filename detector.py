from ultralytics import YOLO
import cv2
import supervision as sv
import time
import numpy as np

# Caminho do vídeo e modelo
video = "videos/tomatesdef.mp4"
model = YOLO(r"C:\Users\danil\Downloads\YoloProject\models\videostomatoesteiran3\weights\best.pt").to(device="cuda")
out_path = "output/tomates2rec.mp4"

#C:\Users\danil\Downloads\YoloProject\models\ModeloLargeMooVisionYolo11\weights\best.pt
class_map = {
    0: "De vez",
    1: "Maduro",
    2: "Verde",
    3: "Fundo",
}


# Inicializa a captura de vídeo
cap = cv2.VideoCapture(video)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
framecount = 0

# Configurações do tracker e anotadores
tracker = sv.ByteTrack()
RESIZE_SCALE = 1
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
# polygon_annotator = sv.PolygonAnnotator()
pontos_x = [33, 27, 404, 397]
pontos_y = [724, 752, 742, 714]

x_min = int(min(pontos_x)* RESIZE_SCALE)
y_min = int(min(pontos_y)* RESIZE_SCALE)
x_max = int(max(pontos_x)* RESIZE_SCALE)
y_max = int(max(pontos_y)* RESIZE_SCALE)

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
    result = model(image_slice,conf=0.5,classes=19)[0]
    return sv.Detections.from_ultralytics(result)

# Inicialize o slicer (exemplo com fatias 320x320 e 20% de sobreposição)
slicer = sv.InferenceSlicer(
    callback=callback,
    slice_wh=(340, 340),
    # overlap_ratio_wh=None,
    # overlap_wh=(0.2, 0.2)
)

def esta_dentro_retangulo(bbox, x_min, y_min, x_max, y_max):
    """
    Verifica se o centro da bbox está dentro do retângulo
    bbox: [x1, y1, x2, y2]
    """
    centro_x = (bbox[0] + bbox[2]) / 2
    centro_y = (bbox[1] + bbox[3]) / 2
    
    return x_min <= centro_x <= x_max and y_min <= centro_y <= y_max

totaltomates = 0
verde = 0
devez = 0
maduros = 0
# Loop principal
while cap.isOpened():
    start_time = time.time()  # Início para cálculo de FPS
    ret, frame = cap.read()
    if not ret:
        break
    framecount += 1
    # Redimensiona o frame
    # frame = cv2.resize(frame, (width, height))
    # frame = cv2.resize(frame, (384, 384))
    # detections = slicer(frame)
    result = model(frame,conf=0.55)[0]#,classes=19
    detections = sv.Detections.from_ultralytics(result)
    if len(detections) > 0:
        detections = tracker.update_with_detections(detections)
        for i, (bbox, tracker_id, class_id) in enumerate(zip(detections.xyxy, detections.tracker_id, detections.class_id)):
            if esta_dentro_retangulo(bbox, x_min, y_min, x_max, y_max):
                # Se o ID ainda não foi contado, adiciona à contagem
                if tracker_id not in counted_ids:
                    if class_id == 0:
                        devez += 1
                    elif class_id == 1:
                        maduros += 1
                    else:
                        verde += 1
                    counted_ids.add(tracker_id)
                    totaltomates += 1
    else:
        print("No Class Detected")
    #classes=19 para vacas
    labels = [
    f"{class_map.get(class_id, f'Classe {class_id}')} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    totais2 = len(detections)
    item_count = len(counted_ids)
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, 
        detections=detections, 
        labels=labels
    )

    # Cria uma cópia para exibição com FPS e contagem
    cv2.putText(annotated_frame, f"Verdes: {verde}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"De vez: {devez}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Maduros: {maduros}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"Total Detectado: {item_count}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # cv2.putText(annotated_frame, f"Total de Vacas: {totaltomates}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    # Calcula e exibe FPS
    end_time = time.time()
    fps_display = round(1 / (end_time - start_time), 2)
    # cv2.putText(annotated_frame, f"Criancas Detectadas: {criancas_detectadas}", (10, 80), 
                # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    exibir = annotated_frame.copy()
    cv2.putText(exibir, f"FPS: {fps_display}", (300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Exibe a contagem de vacas
       
    # Exibe o frame
    cv2.imshow("video", exibir)
    
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
