from ultralytics import YOLO
import cv2
import supervision as sv
import time

video = "videos/caminhao2.mp4"
model = YOLO(r"C:\Users\danil\Downloads\Atualizado\modelos\CorrentesCaminhaoModelM2\weights\best.pt")
out_path = "outputs/caminhaocorrente04.mp4"
cap = cv2.VideoCapture(video)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
framecount = 0
#tracker = sv.ByteTrack() #se precisar tirar
RESIZE_SCALE = 0.5
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

while cap.isOpened():
    start_time = time.time()  # Start time for FPS calculation
    ret, frame = cap.read()
    if not ret:
        break
    framecount += 1
    frame = cv2.resize(frame, (width, height))
    result = model(frame,conf=0.4)[0]
    detections = sv.Detections.from_ultralytics(result)
    #detections = tracker.update_with_detections(detections) #isso tmb remover se necessário
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    
    # Create a copy for display with FPS
    exibir = annotated_frame.copy()
    # Calculate FPS
    end_time = time.time()
    fps_display = round(1 / (end_time - start_time), 2)

    cv2.putText(exibir, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(annotated_frame)
    print(f"Frame {framecount} de {total_frames}")
    # o Exibir, faz com que ele salve o vídeo, sem precisar mostrar que os FPS 
    cv2.imshow("video", exibir)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()