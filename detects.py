import os
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Configurações
VIDEO_PATH = 'vacas.mp4'
OUTPUT_DIR = 'resultados'
MODEL_PATH = r'C:\Users\danil\Downloads\Atualizado\modelos\ModeloLargeMooVisionYolo11\weights\best.pt'
LINE_POSITION = 0.45  # Posição da linha vertical (0-1)
CROSSING_THRESHOLD = 30  # Aumentado para melhor detecção
RESIZE_SCALE = 0.7
MIN_FRAMES_FOR_CROSSING = 3 # Mínimo de frames para considerar um cruzamento válido

def draw_counter(frame, counts):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Esquerda: {counts['left']}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(frame, f"Direita: {counts['right']}", (w - 200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

def detect_crossings(model, frame, line_x, track_history, crossed_ids, counts):
    results = model.track(frame, persist=True, conf=0.2, iou=0.6, tracker="botsort.yaml")
    
    if results[0].boxes.id is None:
        return crossed_ids, np.array([]), np.array([])  # Retorna listas vazias para boxes e track_ids

    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().numpy()
    current_ids = set()

    for box, track_id in zip(boxes, track_ids):
        current_ids.add(track_id)
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Atualiza histórico de posições
        track_history[track_id].append((cx, cy))
        
        # Mantém apenas os últimos N pontos no histórico
        if len(track_history[track_id]) > 20:
            track_history[track_id] = track_history[track_id][-20:]

        # Verifica cruzamento apenas se tiver histórico suficiente
        if len(track_history[track_id]) >= MIN_FRAMES_FOR_CROSSING and track_id not in crossed_ids:
            # Pega a posição mais antiga no histórico
            first_cx, first_cy = track_history[track_id][0]
            
            # Verifica se cruzou da esquerda para direita
            if first_cx < line_x - CROSSING_THRESHOLD and cx >= line_x + CROSSING_THRESHOLD:
                counts['right'] += 1
                crossed_ids.add(track_id)
                print(f"Vaca {track_id} → DIREITA (CX: {first_cx:.1f} -> {cx:.1f})")
            
            # Verifica se cruzou da direita para esquerda
            elif first_cx > line_x + CROSSING_THRESHOLD and cx <= line_x - CROSSING_THRESHOLD:
                counts['left'] += 1
                crossed_ids.add(track_id)
                print(f"Vaca {track_id} ← ESQUERDA (CX: {first_cx:.1f} -> {cx:.1f})")

    # Remove IDs que não estão mais no frame
    crossed_ids = crossed_ids.intersection(current_ids)
    return crossed_ids, boxes, track_ids

def process_video():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(OUTPUT_DIR, 'saida.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    track_history = defaultdict(list)
    counts = {'left': 0, 'right': 0}
    crossed_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        h, w = frame.shape[:2]
        line_x = int(w * LINE_POSITION)
        
        # Criar uma cópia do frame para detecção (sem a linha)
        frame_for_detection = frame.copy()

        # Processar detecção no frame sem a linha
        crossed_ids, boxes, track_ids = detect_crossings(model, frame_for_detection, line_x, track_history, crossed_ids, counts)
        
        # Desenhar caixas e IDs no frame de visualização
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Desenhar a linha central e linhas de threshold no frame de visualização, após as caixas
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 2)
        cv2.line(frame, (line_x - CROSSING_THRESHOLD, 0), (line_x - CROSSING_THRESHOLD, h), (0, 255, 255), 1)
        cv2.line(frame, (line_x + CROSSING_THRESHOLD, 0), (line_x + CROSSING_THRESHOLD, h), (0, 255, 255), 1)

        draw_counter(frame, counts)

        out.write(frame)
        cv2.imshow('Contagem de Vacas', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return counts

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    final_counts = process_video()
    print("\nResumo Final:")
    print(f"Total para ESQUERDA: {final_counts['left']}")
    print(f"Total para DIREITA: {final_counts['right']}")

if __name__ == "__main__":
    main()