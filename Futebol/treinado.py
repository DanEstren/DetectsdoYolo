from ultralytics import YOLO
import cv2 
import supervision as sv


model = YOLO("yolo11n.pt").to(device="cpu")
tracker = sv.ByteTrack(minimum_matching_threshold=0.7)
video_path="videos/futebol.mp4"
RESIZE_SCALE = 0.7
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ellipse_annotator = sv.EllipseAnnotator()
triangle_annotator = sv.TriangleAnnotator()
    

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)


frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    person_mask = detections.class_id == 0
    detections = detections[person_mask]
    detections = tracker.update_with_detections(detections)
    
    annotated_frame = ellipse_annotator.annotate(frame, detections)
    annotated_frame = triangle_annotator.annotate(annotated_frame, detections)

    frame_count += 1
    cv2.imshow("Vacas",annotated_frame)
    print(f"Quantidade de Frames: {frame_count} de // {total_frames}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

cap.release()
cv2.destroyAllWindows()