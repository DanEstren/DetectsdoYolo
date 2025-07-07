from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv
from datetime import datetime
import json
import uuid
import os  # Adicionado para manipulação de pastas
import time
from dotenv import load_dotenv
import base64
from groq import Groq

load_dotenv()
# Função para codificar imagem em base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Carregar modelos YOLO
vehicle_detector = YOLO('yolo11s.pt')  # Modelo para detectar veículos
license_plate_detector = YOLO(r'C:\Users\danil\Downloads\Atualizado\modelos\LicensePlateRecognition\weights\best.pt')  # Modelo para detectar placas

# Classes de veículos que queremos rastrear (carro, moto, caminhão, ônibus)
VEHICLE_CLASSES = [2, 3, 5, 7]
RESIZE_SCALE = 0.3

# Configurações do Supervision
byte_tracker = sv.ByteTrack()
byte_tracker2 = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
label_annotator2 = sv.LabelAnnotator()
box_annotator2 = sv.BoxAnnotator()

# Processar vídeo
video_path = 'videos/carsplate.mp4'  # Ou use 0 para webcam
class_map2 = {0: "Placa"}

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Criar pasta para salvar as imagens das placas
output_folder = "frametoocr"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

first_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
unique_id = str(uuid.uuid4())
init_data = {
    "ID da Execucao do Programa": unique_id,
    "Execucao do Programa as": first_time,
}
output_file = "outputs/PlacaCarros.json"
try:
    with open(output_file, 'r') as f:
        output_json = []
except (json.JSONDecodeError, FileNotFoundError):
    output_json = []
                
output_json.append(init_data)

with open(output_file, 'w') as f:
    json.dump(output_json, f, indent=4)

unique_plate = set()

def is_inside_vehicle(plate_box, vehicle_box):
    """Verifica se a bounding box da placa está dentro da bounding box do veículo."""
    px1, py1, px2, py2 = plate_box
    vx1, vy1, vx2, vy2 = vehicle_box
    return px1 >= vx1 and py1 >= vy1 and px2 <= vx2 and py2 <= vy2

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (width, height))
    frame_count += 1
    result = vehicle_detector(frame, conf=0.6, classes=VEHICLE_CLASSES)[0]
    result2 = license_plate_detector(frame, conf=0.3)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections2 = sv.Detections.from_ultralytics(result2)
    detections = byte_tracker.update_with_detections(detections)
    detections2 = byte_tracker2.update_with_detections(detections2)

    for idx, (plate_tracker_id, plate_conf, plate_box) in enumerate(zip(detections2.tracker_id, detections2.confidence, detections2.xyxy)):
        if plate_tracker_id is None or plate_tracker_id in unique_plate:
            continue
        # Procurar veículo correspondente
        vehicle_id = None
        for vehicle_tracker_id, vehicle_box in zip(detections.tracker_id, detections.xyxy):
            if vehicle_tracker_id is None:
                continue
            if is_inside_vehicle(plate_box, vehicle_box):
                vehicle_id = vehicle_tracker_id
                break
        if vehicle_id is not None:  # Só registrar se a placa estiver dentro de um veículo
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            unique_plate.add(plate_tracker_id)
            # Recortar e salvar a imagem da placa
            x1, y1, x2, y2 = map(int, plate_box)  # Converter coordenadas para inteiros
            plate_image = frame[y1:y2, x1:x2]  # Recortar a região da placa
            plate_filename = os.path.join(output_folder, f"plate_{plate_tracker_id}_{current_time.replace(':', '-')}.jpg")
            cv2.imwrite(plate_filename, plate_image)  # Salvar a imagem
            base64_image = encode_image(plate_filename)
            print("Imagem Transformada, tentando API")
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract the text from the image, i only want the letters and number form the car carplate, nothing more, don't say anything else"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                )
                content = chat_completion.choices[0].message.content
                print(f"Sucesso na API em {current_time}: {content}")
                time.sleep(3)      
            except:
                print("Falhou na API")
                content= "Falha na API" 
                time.sleep(3)       
            entry = {
                "ID do Carro": int(vehicle_id),
                "ID da Placa": int(plate_tracker_id),
                "confidence": float(plate_conf),
                "Plate Name": content,
                "exit_timestamp": current_time
            }
            output_json.append(entry)

            print("Removendo o Arquivo")
            os.remove(plate_filename)
            print("Arquivo Removido")


    # Escrever no arquivo após processar todas as placas no frame
    with open(output_file, 'w') as f:
        json.dump(output_json, f, indent=4)

    labels = [
        f"{vehicle_detector.names[class_id] if hasattr(vehicle_detector, 'names') else f'Class_{class_id}'} (ID: {tracker_id})"
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)] 

    # Criar rótulos para o segundo modelo (detections2)
    labels2 = [
        f"{class_map2.get(class_id, license_plate_detector.names[class_id])} ID:({tracker_id})"
        for class_id, tracker_id in zip(detections2.class_id, detections2.tracker_id)
    ] if detections2.class_id is not None else []

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    annotated_frame = box_annotator2.annotate(scene=annotated_frame, detections=detections2)
    annotated_frame = label_annotator2.annotate(scene=annotated_frame, detections=detections2, labels=labels2)
    cv2.putText(annotated_frame, f"Placas Detectadas: {len(detections2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar frame (opcional para debug)
    cv2.imshow('Detector de Placas', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()