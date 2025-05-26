import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Load the custom YOLO model
model = YOLO(r"C:\Users\danil\Downloads\Atualizado\modelos\CoffeCherryYoloL\weights\best.pt")

# Load the video
video_path = r"C:\Users\danil\Downloads\Atualizado\videos\cafesplantaszap.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the output video writer
output_path = r"C:\Users\danil\Downloads\Atualizado\videos\output_annotated_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define the callback function for InferenceSlicer
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

# Initialize Supervision annotators
slicer = sv.InferenceSlicer(callback=callback)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Process video frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    detections = slicer(frame)

    # Annotate frame
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Write the annotated frame to the output video
    cv2.imshow("Cafe",annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.write(annotated_frame)

    frame_count += 1
    print(f"Processed frame {frame_count}/{total_frames}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved to: {output_path}")