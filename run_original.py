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

        # CSV output
        self.csv_path = "detections.csv"
        self.csv_file = open(self.csv_path, "w")
        self.csv_file.write(
            "frame_id,track_id,label,confidence,xmin,ymin,xmax,ymax\n"
        )
        self.csv_file.flush()

    def write_csv(self, line):
        self.csv_file.write(line + "\n")
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

# -----------------------------------------------------------------------------------------------
# Callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Frame counter
    user_data.increment()
    frame_id = user_data.get_count()

    # Frame info
    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for det in detections:
        label = det.get_label()
        confidence = det.get_confidence()
        bbox = det.get_bbox()

        # Tracking ID (if exists)
        track_id = 0
        track = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()

        # Convert normalized bbox → pixels
        xmin = int(bbox.xmin() * width)
        ymin = int(bbox.ymin() * height)
        xmax = int(bbox.xmax() * width)
        ymax = int(bbox.ymax() * height)

        # Write CSV
        user_data.write_csv(
            f"{frame_id},{track_id},{label},{confidence:.4f},"
            f"{xmin},{ymin},{xmax},{ymax}"
        )

        # Optional drawing
        if frame is not None:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    finally:
        user_data.close()
