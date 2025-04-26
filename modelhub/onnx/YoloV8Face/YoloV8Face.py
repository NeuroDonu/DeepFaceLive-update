from pathlib import Path
from typing import List, Tuple
import numpy as np
from xlib import math as lib_math
from xlib.image import ImageProcessor
from xlib.onnxruntime import (
    InferenceSession_with_device,
    ORTDeviceInfo,
    get_available_devices_info,
)

class YoloV8Face:
    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        if device_info not in YoloV8Face.get_available_devices():
            raise Exception(f"device_info {device_info} is not available for YoloV8Face")

        model_path = Path(__file__).parent / "yoloface_8n.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self._sess = InferenceSession_with_device(str(model_path), device_info)
        model_inputs = self._sess.get_inputs()
        if not model_inputs:
            raise RuntimeError("Could not get model inputs from ONNX file.")

        self._input_name = model_inputs[0].name
        shp = model_inputs[0].shape
        try:
            h, w = shp[-2], shp[-1]
            if isinstance(h, int) and isinstance(w, int):
                self._input_height, self._input_width = h, w
            else:
                self._input_height = self._input_width = 640
                #print(f"[YoloV8Face] WARN: Dynamic input shape detected, assuming {self._input_height}x{self._input_width}")

        except Exception as e:
            self._input_height = self._input_width = 640
            print(f"[YoloV8Face] WARN: Error reading input shape ({e}), assuming {self._input_height}x{self._input_width}")

    def extract(
        self,
        img: np.ndarray,
        threshold: float = 0.45,
        nms_threshold: float = 0.5,
        fixed_window: int = 0,
        min_face_size: int = 8,
    ) -> List[List[Tuple[float, float, float, float]]]:
        """Возвращает список лиц для каждого кадра батча."""

        ip = ImageProcessor(img)
        batch, H_orig, W_orig, _ = ip.get_dims()

        # PRE‑PROCESS
        preproc_type = "None"
        img_scale = 1.0
        pad_x = pad_y = 0.0

        if fixed_window:
            preproc_type = "Fixed Window"
            fw = max(32, (fixed_window // 32) * 32)
            img_scale = ip.fit_in(fw, fw, pad_to_target=True, allow_upscale=False)
        else:
            preproc_type = "Pad Divisor 64"
            ip.pad_to_next_divisor(64, 64)
            img_scale = 1.0

        input_tensor = ip.ch(3).to_ufloat32().get_image("NCHW")
        _, _, H_proc, W_proc = input_tensor.shape

        if (H_proc, W_proc) != (self._input_height, self._input_width):
            preproc_type = "Resize Fallback"
            ip = ImageProcessor(img)
            img_scale = min(self._input_width / W_orig, self._input_height / H_orig)
            ip.fit_in(self._input_height, self._input_width, pad_to_target=True, allow_upscale=True)
            input_tensor = ip.ch(3).to_ufloat32().get_image("NCHW")
            _, _, H_proc, W_proc = input_tensor.shape
            # Оставляем предупреждение на случай ошибки ресайза
            if (H_proc, W_proc) != (self._input_height, self._input_width):
                 print(f"[YOLOv8Face] WARN: Fallback resize did not result in exact target dimensions! Got {H_proc}x{W_proc}, expected {self._input_height}x{self._input_width}.")

        if preproc_type == "Fixed Window" or preproc_type == "Resize Fallback":
            scaled_w = W_orig * img_scale
            scaled_h = H_orig * img_scale
            pad_x = (W_proc - scaled_w) / 2.0
            pad_y = (H_proc - scaled_h) / 2.0
        else:
            pad_x = pad_y = 0.0

        # INFERENCE
        out = self._sess.run(None, {self._input_name: input_tensor})[0]

        # Transpose if needed: [B, P, N] -> [B, N, P]
        if out.ndim == 3 and out.shape[1] < out.shape[2]:
            out = out.transpose(0, 2, 1)

        faces_per_batch: List[List[Tuple[float, float, float, float]]] = []
        for i, pred in enumerate(out):  # iterate over batch
            if pred.ndim != 2:
                faces_per_batch.append([])
                continue

            conf_index = 4
            num_coords = 4
            if pred.shape[1] <= conf_index:
                faces_per_batch.append([])
                continue

            mask = pred[:, conf_index] >= threshold
            det = pred[mask]
            if det.size == 0:
                faces_per_batch.append([])
                continue

            coords = det[:, :num_coords]
            conf = det[:, conf_index]

            l, t, r, b = None, None, None, None

            # Attempt 1: cx, cy, w, h
            try:
                cx, cy, w, h = coords.T
                l = cx - w / 2
                t = cy - h / 2
                r = cx + w / 2
                b = cy + h / 2
            except ValueError:
                # Attempt 2: x1, y1, x2, y2
                try:
                    x1, y1, x2, y2 = coords.T
                    l, t, r, b = x1, y1, x2, y2
                except ValueError:
                     # Failed both ways
                     faces_per_batch.append([])
                     continue

            if l is None: # Should not happen if try/except works, but as a safeguard
                 faces_per_batch.append([])
                 continue

            # NMS
            keep = lib_math.nms(l, t, r, b, conf, nms_threshold)
            if len(keep) == 0:
                 faces_per_batch.append([])
                 continue

            l, t, r, b = l[keep], t[keep], r[keep], b[keep]

            # Inverse Transform to Original Coords (NO PAD SUBTRACTION)
            l_orig = l / img_scale
            r_orig = r / img_scale
            t_orig = t / img_scale
            b_orig = b / img_scale

            # Clipping
            l_clipped = np.clip(l_orig, 0, W_orig)
            r_clipped = np.clip(r_orig, 0, W_orig)
            t_clipped = np.clip(t_orig, 0, H_orig)
            b_clipped = np.clip(b_orig, 0, H_orig)

            # Filter by size & Collect results
            faces: List[Tuple[float, float, float, float]] = []
            for li, ti, ri, bi in zip(l_clipped, t_clipped, r_clipped, b_clipped):
                face_w, face_h = ri - li, bi - ti
                if face_w >= min_face_size and face_h >= min_face_size:
                    faces.append((float(li), float(ti), float(ri), float(bi)))

            # Expand to fixed window (if needed)
            if fixed_window and faces:
                 half = fixed_window / 2.0
                 expanded: List[Tuple[float, float, float, float]] = []
                 for li, ti, ri, bi in faces:
                     cx_i, cy_i = (li + ri) / 2.0, (ti + bi) / 2.0
                     nl = max(cx_i - half, 0.0)
                     nt = max(cy_i - half, 0.0)
                     nr = min(cx_i + half, W_orig)
                     nb = min(cy_i + half, H_orig)
                     expanded.append((nl, nt, nr, nb))
                 faces = expanded

            faces_per_batch.append(faces)

        return faces_per_batch
