from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import os
import sys
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
# Helper: extract model name from --hef argument
# -----------------------------------------------------------------------------------------------
def get_model_name_from_args():
    if "--hef" in sys.argv:
        idx = sys.argv.index("--hef") + 1
        hef_path = Path(sys.argv[idx])
        return hef_path.stem
    return "model"
    
def get_input_name_from_args():
    if "--input" in sys.argv:
        idx = sys.argv.index("--input") + 1
        input_path = Path(sys.argv[idx])
        return input_path.stem
    return "model"

# -----------------------------------------------------------------------------------------------
# User callback data class
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self, model_name, input_name):
        super().__init__()

        self.model_name = model_name
        self.input_name = input_name

        # CSV output
        csv_dir = Path(f"doc/p/{self.model_name}")
        csv_dir.mkdir(parents=True, exist_ok=True)

        self.csv_file = open(
            csv_dir / f"{self.model_name}_{input_name.split('/')[-1][:-4]}_detections_images.csv", "w"
        )
        self.csv_file.write(
            "image_id,label,confidence,xmin,ymin,xmax,ymax\n"
        )
        self.csv_file.flush()

        # Annotated image output directory
        self.output_dir = Path("annotated_images") / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Model name      : {self.model_name}")
        print(f"[INFO] CSV output      : {self.csv_file.name}")
        print(f"[INFO] Image output dir: {self.output_dir}")

    def write_csv(self, line):
        self.csv_file.write(line + "\n")
        self.csv_file.flush()

    def image_path(self, image_id):
        return self.output_dir / f"{self.model_name}_{input_name.split('/')[-1][:-4]}_{image_id:05d}.jpg"

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

    # Force frame extraction
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

        # Normalized bbox (for CSV)
        xmin_n = bbox.xmin()
        ymin_n = bbox.ymin()
        xmax_n = bbox.xmax()
        ymax_n = bbox.ymax()

        user_data.write_csv(
            f"{image_id},{label},{confidence:.4f},"
            f"{xmin_n},{ymin_n},{xmax_n},{ymax_n}"
        )

        # Pixel bbox (for drawing)
        xmin = int(xmin_n * width)
        ymin = int(ymin_n * height)
        xmax = int(xmax_n * width)
        ymax = int(ymax_n * height)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (xmin, max(0, ymin - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Save annotated image
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out_path = user_data.image_path(image_id)

    if cv2.imwrite(str(out_path), frame):
        print(f"[OK] Saved {out_path}")
    else:
        print(f"[ERROR] Failed to save {out_path}")

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    # Extract model name from --hef
    model_name = get_model_name_from_args()
    input_name = get_input_name_from_args()

    user_data = user_app_callback_class(model_name,input_name)

    # REQUIRED for frame extraction
    user_data.use_frame = True

    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    finally:
        user_data.close()
