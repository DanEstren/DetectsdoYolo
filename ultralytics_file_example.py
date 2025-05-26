import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

# Define constants
COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)

# Hardcoded paths and settings
SOURCE_VIDEO_PATH = r"C:\Users\danil\Downloads\Atualizado\videos\caixamercado.mp4"
ZONE_CONFIGURATION_PATH = r"C:\Users\danil\Downloads\Atualizado\checkout\config.json"
WEIGHTS = r"C:\Users\danil\Downloads\Atualizado\yolo11x.pt"
DEVICE = "cuda"
CONFIDENCE = 0.3
IOU = 0.7
CLASSES = [0]  # Empty list to track all classes; modify if specific classes are needed

output_path = r"C:\Users\danil\Downloads\Atualizado\outputs\videomercado.mp4"

cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))



def main() -> None:
    # Load the custom YOLO model
    model = YOLO(WEIGHTS)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    frames_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # Load zone configuration
    polygons = load_zones_config(file_path=ZONE_CONFIGURATION_PATH)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]

    # Process video frames
    for frame in frames_generator:
        results = model(frame, verbose=False, device=DEVICE, conf=CONFIDENCE)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[find_in_list(detections.class_id, CLASSES)]
        detections = detections.with_nms(threshold=IOU)
        detections = tracker.update_with_detections(detections)

        annotated_frame = frame.copy()

        for idx, zone in enumerate(zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        out.write(annotated_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()