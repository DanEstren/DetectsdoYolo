from flask import Flask, Response, render_template
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Configurações
MODEL_PATH = 'yolo11n.pt'  # Modelo pré-treinado YOLO11n
CAMERA_INDEX = 0  # Índice da câmera (0 para webcam padrão)
RESIZE_SCALE = 0.7  # Escala para redimensionar o frame

# Inicializar o modelo YOLO
model = YOLO(MODEL_PATH)

# Função para gerar cores distintas para cada classe
def get_color_for_class(class_id, num_classes=80):
    # Gerar cores baseadas em um espaço de cores HSV
    hue = (class_id * (360 // num_classes)) % 360  # Distribuir tons
    saturation = 1.0
    value = 1.0
    # Converter HSV para RGB (valores entre 0 e 255)
    hsv_color = np.array([[[hue, saturation * 255, value * 255]]], dtype=np.uint8)
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
    return tuple(int(c) for c in rgb_color)  # Retorna (R, G, B)

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Redimensionar o frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
        frame = cv2.resize(frame, (width, height))

        # Realizar detecção com YOLO
        results = model.track(frame, persist=True, conf=0.3, iou=0.6, tracker="botsort.yaml")

        # Desenhar bounding boxes e nomes das classes
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                class_id = class_ids[i]
                class_name = model.names[class_id]  # Nome da classe (ex.: person, bottle)

                # Obter cor específica para a classe
                color = get_color_for_class(class_id)

                # Desenhar bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Exibir apenas o nome da classe
                cv2.putText(frame, class_name, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Codificar o frame como JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Enviar o frame como parte do stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)