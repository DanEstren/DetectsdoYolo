from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from datetime import datetime
import json
import os
import time

# Initialize YOLO model, video capture, and supervision components
model = YOLO(r"C:\Users\danil\Downloads\Atualizado\modelos\Mercado6ModelS\weights\best.pt")
cap = cv2.VideoCapture("videos/mercado6.mp4")
out_path = "outputs/mercado6final.mp4"
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# Configurações do vídeo de saída
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# Initialize JSON data structure and set output file
output_json = []
output_file = "detected_products.json"

# Set to keep track of unique product IDs
unique_product_ids = set()
# Counter for total detected products
total_products = 0
# Dictionary to track frame counts for each tracker ID
tracker_frame_counts = {}

counting_area = np.array([(453, 332), (875, 324), (880, 550), (444, 558)])
counting_area2 = np.array([(635, 318), (646, 537), (746, 536), (731, 310)])

while cap.isOpened():
    start_time = time.time() 
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform detection
    result = model(frame, conf=0.6)[0]
    detections = sv.Detections.from_ultralytics(result)
    
    # Update detections with tracker to get unique IDs
    detections = tracker.update_with_detections(detections)
    
    # Get current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create custom labels combining class name and tracker ID
    labels = [
        f"{model.names[class_id] if hasattr(model, 'names') else f'Class_{class_id}'} (ID: {tracker_id})"
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
    ]
    
    # Update frame counts for detected tracker IDs
    current_tracker_ids = set(detections.tracker_id)
    for tracker_id in current_tracker_ids:
        tracker_frame_counts[tracker_id] = tracker_frame_counts.get(tracker_id, 0) + 1
    
    # Process each detection for JSON saving
    for det in zip(detections.xyxy, detections.class_id, detections.tracker_id, detections.confidence):
        bbox, class_id, tracker_id, confidence = det
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        center_point = (int(x_center), int(y_center))

        cv2.circle(frame, center=center_point, radius=5, color=(0, 0, 255), thickness=-1)

        class_name = model.names[class_id] if hasattr(model, 'names') else f"Class_{class_id}"
        
        # Check if detection is in both counting areas and has been detected for more than 2 frames
        if (cv2.pointPolygonTest(counting_area, center_point, False) > 0 and 
            cv2.pointPolygonTest(counting_area2, center_point, False) > 0 and
            tracker_frame_counts.get(tracker_id, 0) > 4):
            if tracker_id not in unique_product_ids:
                unique_product_ids.add(tracker_id)
                detection_data = {
                    "timestamp": current_time,
                    "tracker_id": int(tracker_id),
                    "class_name": class_name,
                    "confidence": float(confidence)
                }
                output_json.append(detection_data)
                total_products += 1
    
    # Remove tracker IDs that are no longer detected
    tracker_frame_counts = {tid: count for tid, count in tracker_frame_counts.items() if tid in current_tracker_ids}
    
    # Annotate frame with boxes and custom labels
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.polylines(annotated_frame, pts=[counting_area], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.polylines(annotated_frame, pts=[counting_area2], isClosed=True, color=(255, 0, 0), thickness=3)
    
    # Display counter on frame
    counter_text = f"Total Products Detected: {total_products}"
    cv2.putText(
        annotated_frame,
        counter_text,
        (10, 30),  # Position at top-left corner
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # Font scale
        (0, 255, 0),  # Green color
        2  # Thickness
    )
    # Write frame to output video
    out.write(annotated_frame)
    exibir = annotated_frame.copy()
    # Calcula e exibe FPS
    end_time = time.time()
    fps_display = round(1 / (end_time - start_time), 2)
    cv2.putText(exibir, f"FPS: {fps_display}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show annotated frame
    cv2.imshow("janela", exibir)
    
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save detection data to JSON file
with open(output_file, 'w') as f:
    json.dump(output_json, f, indent=4)

print(f"Detection data saved to {output_file}")
print(f"Total unique products detected: {total_products}")