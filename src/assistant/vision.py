from __future__ import annotations

import cv2

from .config import VisionCfg


class Webcam:
    """Grabs a single fresh frame from the webcam on demand.

    Opens the device per-grab so the camera light isn't on when not in use,
    and so audio capture isn't competing with v4l2 buffers continuously.
    """

    def __init__(self, cfg: VisionCfg):
        self.cfg = cfg

    def grab_jpeg(self) -> bytes | None:
        cap = cv2.VideoCapture(self.cfg.device_index)
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            # Many webcams need a few throwaway reads to deliver a sharp frame.
            for _ in range(3):
                cap.read()
            ok, frame = cap.read()
            if not ok:
                return None
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
            return buf.tobytes() if ok else None
        finally:
            cap.release()
