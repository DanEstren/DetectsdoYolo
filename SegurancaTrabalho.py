from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np

# Carregar os modelos
model = YOLO(r"C:\Users\danil\Downloads\Atualizado\modelos\SegurancanotrabmodelS\weights\best.pt").to(device="cuda")
model2 = YOLO(r"C:\Users\danil\Downloads\Atualizado\modelos\FallDetector25SV2\weights\best.pt").to(device="cuda")

# Configurar captura de vídeo e saída
cap = cv2.VideoCapture("videos/TrabInd2.mp4")
out_path = "outputs/SegurancaNoTrab2.mp4"
tracker = sv.ByteTrack()
RESIZE_SCALE = 0.6
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# Mapeamento de classes personalizado
class_map = {0: "Capacete", 1: "Sem Capacete", 2: "Sem Colete", 3: "Colete"}
class_map2 = {0: "Caida", 1: "Pessoa"}

# Definir uma paleta de cores personalizada
custom_colors1 = [
    sv.Color(r=255, g=0, b=255),  # Vermelho para a classe 0 (Caida)
    sv.Color(r=0, g=255, b=0),  # Verde para a classe 1 (Pessoa)
    sv.Color(r=0,g=0,b=255),
    sv.Color(r=0,g=255,b=255),
]
custom_colors2 = [
    sv.Color(r=255, g=0, b=0),  # Vermelho para a classe 0 (Caida)
    sv.Color(r=0, g=255, b=0),  # Verde para a classe 1 (Pessoa)
]


# Criar um ColorPalette personalizado
color_palette = sv.ColorPalette(colors=custom_colors1)
color_palette2 = sv.ColorPalette(colors=custom_colors2)

# Configurar anotadores
box_annotator = sv.BoxAnnotator(color=color_palette)
box_annotator2 = sv.BoxAnnotator(color=color_palette2)
label_annotator = sv.LabelAnnotator(
    text_position=sv.Position.CENTER,
    color=color_palette,
    text_color=sv.Color.WHITE,
    text_scale=0.5,
    text_thickness=1,
    text_padding=2
)

label_annotator2 = sv.LabelAnnotator(
    color=color_palette2,
    text_color=sv.Color.BLACK,
    text_scale=0.5,
    text_thickness=1,
    text_padding=2
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    
    # Realizar inferência com os dois modelos
    result = model(frame,conf=0.6)[0]
    result2 = model2(frame, classes=[0], conf=0.7)[0]
    
    # Converter resultados para detecções do Supervision
    detections = sv.Detections.from_ultralytics(result)
    detections2 = sv.Detections.from_ultralytics(result2)
    
    # Atualizar tracker para detecções do segundo modelo
    detections2 = tracker.update_with_detections(detections2)
    quantidadefall = len(detections2.tracker_id) if detections2.tracker_id is not None else 0

    # Criar rótulos para o primeiro modelo (detections)
    labels = [
        f"{class_map.get(class_id, model.names[class_id])} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ] if detections.class_id is not None else []

    # Criar rótulos para o segundo modelo (detections2)
    labels2 = [
        f"{class_map2.get(class_id, model2.names[class_id])} {confidence:.2f}"
        for class_id, confidence in zip(detections2.class_id, detections2.confidence)
    ] if detections2.class_id is not None else []

    # Anotar a imagem com caixas e rótulos
    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = box_annotator2.annotate(annotated_frame, detections2)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
    annotated_frame = label_annotator2.annotate(annotated_frame, detections2, labels=labels2)

    # Adicionar texto de aviso se houver quedas
    if quantidadefall > 0:
        cv2.putText(
            annotated_frame,
            f"Aviso! Pessoa Caida: {quantidadefall}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Escrever frame no vídeo de saída e exibir
    out.write(annotated_frame)
    cv2.imshow("janela", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
out.release()
cap.release()
cv2.destroyAllWindows()