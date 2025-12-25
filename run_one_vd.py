from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import os
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    app_callback_class
)
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import (
    GStreamerDetectionApp
)

# -----------------------------------------------------------------------------------------------
# User callback data class
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

        # CSV
        self.csv_file = open(f"doc/p/yolov11n_detections_images.csv", "w")
        self.csv_file.write(
            "image_id,label,confidence,xmin,ymin,xmax,ymax\n"
        )
        self.csv_file.flush()

        # Output images
        self.output_dir = Path("annotated_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(self, line):
        self.csv_file.write(line + "\n")
        self.csv_file.flush()

    def image_path(self, image_id):
        return self.output_dir / f"yolov11n_image_{image_id:05d}.jpg"

    def close(self):
        self.csv_file.close()

# -----------------------------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Image counter
    user_data.increment()
    image_id = user_data.get_count()

    format, width, height = get_caps_from_pad(pad)

    # 🔴 FORCE frame extraction
    frame = get_numpy_from_buffer(buffer, format, width, height)
    if frame is None:
        print("[WARN] Frame extraction failed")
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for det in detections:
        label = det.get_label()
        confidence = det.get_confidence()
        bbox = det.get_bbox()

        xmin = bbox.xmin()#int(bbox.xmin() * width)
        ymin = bbox.ymin()#int(bbox.ymin() * height)
        xmax = bbox.xmax()#int(bbox.xmax() * width)
        ymax = bbox.ymax()#int(bbox.ymax() * height)

        user_data.write_csv(
            f"{image_id},{label},{confidence:.4f},"
            f"{xmin},{ymin},{xmax},{ymax}"
        )
        
        xmin = int(bbox.xmin() * width)
        ymin = int(bbox.ymin() * height)
        xmax = int(bbox.xmax() * width)
        ymax = int(bbox.ymax() * height)

        # Draw annotation
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (xmin, max(0, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # 🔴 SAVE IMAGE EXPLICITLY
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out_path = user_data.image_path(image_id)
    ok = cv2.imwrite(str(out_path), frame)

    if not ok:
        print(f"[ERROR] Failed to save image {out_path}")
    else:
        print(f"[OK] Saved {out_path}")

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    user_data = user_app_callback_class()

    # 🔴 REQUIRED
    user_data.use_frame = True

    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    finally:
        user_data.close()
