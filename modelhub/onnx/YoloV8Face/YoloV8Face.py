from pathlib import Path
from typing import List, Tuple
import numpy as np
from xlib import math as lib_math
from xlib.image import ImageProcessor
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)

# --- DEBUG FLAG ---
# Установите в True, чтобы включить подробную отладочную печать
DEBUG_YOLO = False # Оставляем включенным для проверки

class YoloV8Face:
    # ... (init и get_available_devices остаются без изменений) ...
    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        """Returns a list of available devices supported by ONNX Runtime."""
        return get_available_devices_info()

    def __init__(self, device_info : ORTDeviceInfo ):
        if device_info not in YoloV8Face.get_available_devices():
            raise Exception(f'device_info {device_info} is not in available devices for YoloV8Face')

        model_path = Path(__file__).parent / 'yoloface_8n.onnx' # Adjust filename if needed
        if not model_path.exists():
             raise FileNotFoundError(f"Model file not found at {model_path}")

        self._sess = sess = InferenceSession_with_device(str(model_path), device_info)
        model_inputs = sess.get_inputs()
        if not model_inputs:
            raise RuntimeError("Could not get model inputs from ONNX file.")

        self._input_name = model_inputs[0].name
        self._input_shape = model_inputs[0].shape

        try:
            if isinstance(self._input_shape[-2], int) and isinstance(self._input_shape[-1], int):
                 self._input_height = self._input_shape[-2]
                 self._input_width = self._input_shape[-1]
            else:
                 if DEBUG_YOLO: print(f"[YoloV8Face Debug] Warning: Model input shape dimensions appear dynamic ({self._input_shape}). Using default 640x640.")
                 self._input_height = 640
                 self._input_width = 640
        except (TypeError, IndexError):
             if DEBUG_YOLO: print(f"[YoloV8Face Debug] Warning: Could not reliably determine input H/W from shape {self._input_shape}. Using default 640x640.")
             self._input_height = 640
             self._input_width = 640

        if DEBUG_YOLO: print(f"[YoloV8Face Debug] Initialized with model: {model_path}, device: {device_info}, input shape: {self._input_shape}, target size: {self._input_width}x{self._input_height}")


    def extract(self, img: np.ndarray, threshold: float = 0.45, nms_threshold: float = 0.5, fixed_window: int = 0, min_face_size: int = 8) -> List[List[Tuple[float, float, float, float]]]:
        # ... (Preprocessing остается без изменений) ...
        ip = ImageProcessor(img)
        batch_size, H_orig, W_orig, _ = ip.get_dims()
        img_scale = 1.0
        target_height = self._input_height
        target_width = self._input_width
        if fixed_window > 0 and (fixed_window != target_width or fixed_window != target_height):
             if DEBUG_YOLO: print(f"[YoloV8Face Debug] Предупреждение: Параметр fixed_window={fixed_window} проигнорирован. Модель требует {target_width}x{target_height}.")
        if DEBUG_YOLO: print(f"[YoloV8Face Debug] Original image size: {W_orig}x{H_orig}. Target size: {target_width}x{target_height}")
        img_scale = ip.fit_in(target_width, target_height, pad_to_target=True, allow_upscale=True)
        if DEBUG_YOLO: print(f"[YoloV8Face Debug] Image scale factor after fit_in: {img_scale}")
        input_tensor = ip.ch(3).to_ufloat32().get_image('NCHW')
        _, _, H_proc, W_proc = input_tensor.shape
        if DEBUG_YOLO: print(f"[YoloV8Face Debug] Preprocessed tensor shape: {input_tensor.shape}")
        if H_proc != target_height or W_proc != target_width:
             # ... (Fallback resize logic) ...
             if DEBUG_YOLO: print(f"[YoloV8Face Debug] Warning: Preprocessed shape ({H_proc}, {W_proc}) != target ({target_height}, {target_width}). Retrying with resize.")
             ip = ImageProcessor(img); img_scale = min(target_width / W_orig, target_height / H_orig); ip.resize(target_width, target_height, interpolation=ImageProcessor.Interpolation.LINEAR); input_tensor = ip.ch(3).to_ufloat32().get_image('NCHW'); _, _, H_proc, W_proc = input_tensor.shape
             if DEBUG_YOLO: print(f"[YoloV8Face Debug] Tensor shape after resize fallback: {input_tensor.shape}, New scale: {img_scale}")
             if H_proc != target_height or W_proc != target_width: raise RuntimeError(f"Failed to preprocess image to required size ({target_height}, {target_width}). Final shape: ({H_proc}, {W_proc})")

        # --- Inference ---
        if DEBUG_YOLO: print(f"[YoloV8Face Debug] Running inference...")
        outputs = self._sess.run(None, {self._input_name: input_tensor})
        if DEBUG_YOLO: print(f"[YoloV8Face Debug] Inference done. Output type: {type(outputs)}, len: {len(outputs)}")
        if isinstance(outputs, list) and len(outputs) > 0:
             if DEBUG_YOLO: print(f"[YoloV8Face Debug] Raw model output[0] shape: {outputs[0].shape}")
        else:
             if DEBUG_YOLO: print(f"[YoloV8Face Debug] Unexpected output format from model.")
             return [[] for _ in range(batch_size)]

        # --- Postprocessing ---
        preds = outputs[0] # Shape (1, 20, 8400)

        # Transpose: (N, Features, Boxes) -> (N, Boxes, Features)
        num_features_expected_min = 5 # cx, cy, w, h, conf
        # Модель уже выводит (1, 20, 8400), нужно транспонировать в (1, 8400, 20)
        if len(preds.shape) == 3 and preds.shape[1] < preds.shape[2] and preds.shape[1] >= num_features_expected_min:
             if DEBUG_YOLO: print(f"[YoloV8Face Debug] Transposing model output from {preds.shape}...")
             preds = preds.transpose(0, 2, 1) # Теперь shape (1, 8400, 20)
             if DEBUG_YOLO: print(f"[YoloV8Face Debug] Transposed output shape: {preds.shape}")
        elif len(preds.shape) == 3 and preds.shape[2] < preds.shape[1] and preds.shape[2] >= num_features_expected_min:
             if DEBUG_YOLO: print(f"[YoloV8Face Debug] Output shape {preds.shape} already seems to be (N, Boxes, Features). No transpose needed.")
             # Already in the correct format (N, Boxes, Features), e.g. (1, 8400, 20)
             pass
        else:
             # If it was already (1, 8400, 20), this check should pass
             if len(preds.shape) < 3 or preds.shape[-1] < num_features_expected_min:
                  if DEBUG_YOLO: print(f"[YoloV8Face Debug] ERROR: Model output has unexpected shape or feature dimension size: {preds.shape}. Expected at least {num_features_expected_min} features.")
                  raise ValueError(f"Model output has unexpected shape or feature dimension size: {preds.shape}.")


        faces_per_batch = []
        for i in range(batch_size):
            pred = preds[i] # Shape (8400, 20)
            if DEBUG_YOLO: print(f"\n--- [YoloV8Face Debug] Processing Image {i} ---")
            if DEBUG_YOLO: print(f"Pred shape for image {i}: {pred.shape}")

            # --- <<< ОТЛАДКА СЫРЫХ ДАННЫХ (ОСТАВЛЯЕМ) >>> ---
            if DEBUG_YOLO and pred.shape[0] > 0:
                 num_rows_to_print = min(5, pred.shape[0])
                 num_cols_to_print = min(6, pred.shape[-1])
                 print(f"Raw pred data sample (first {num_rows_to_print} rows, first {num_cols_to_print} cols):")
                 print(pred[:num_rows_to_print, :num_cols_to_print])
            # --- <<< КОНЕЦ ОТЛАДКИ СЫРЫХ ДАННЫХ >>> ---

            # Filter by confidence threshold (Используем индекс 4, как подтвердилось)
            conf_index = 4
            if DEBUG_YOLO: print(f"Using confidence index: {conf_index}. Threshold: {threshold}")
            confident_detections = pred[pred[..., conf_index] >= threshold]
            if DEBUG_YOLO: print(f"Detections above threshold ({confident_detections.shape[0]}):")
            if DEBUG_YOLO and confident_detections.shape[0] > 0 and confident_detections.shape[0] < 10:
                print(confident_detections[:, :min(6, confident_detections.shape[-1])]) # Print first few columns

            if confident_detections.shape[0] == 0:
                faces_per_batch.append([])
                if DEBUG_YOLO: print(f"No confident detections for image {i}.")
                continue

            # --- <<< ИЗВЛЕЧЕНИЕ КООРДИНАТ КАК CX, CY, W, H >>> ---
            # ИСПОЛЬЗУЕМ ИНДЕКСЫ 0, 1, 2, 3
            cx_idx, cy_idx, w_idx, h_idx = 0, 1, 2, 3
            if DEBUG_YOLO: print(f"Using coordinate indices: cx={cx_idx}, cy={cy_idx}, w={w_idx}, h={h_idx}")

            cx = confident_detections[:, cx_idx]
            cy = confident_detections[:, cy_idx]
            w = confident_detections[:, w_idx]
            h = confident_detections[:, h_idx]
            score = confident_detections[:, conf_index]

            # --- <<< КОНВЕРТАЦИЯ CX, CY, W, H -> L, T, R, B >>> ---
            if DEBUG_YOLO: print("Converting cx, cy, w, h -> l, t, r, b")
            l = cx - w / 2
            t = cy - h / 2
            r = cx + w / 2
            b = cy + h / 2
            # --- <<< КОНЕЦ КОНВЕРТАЦИИ >>> ---

            if DEBUG_YOLO and len(l) > 0:
                 print(f"Boxes before NMS (cx,cy,w,h -> l,t,r,b) (first 5):")
                 for k in range(min(5, len(l))):
                      print(f"  Box {k}: l={l[k]:.2f}, t={t[k]:.2f}, r={r[k]:.2f}, b={b[k]:.2f}, score={score[k]:.3f}")

            # Apply Non-Maximum Suppression (NMS)
            if DEBUG_YOLO: print(f"Running NMS with threshold: {nms_threshold}")
            keep_indices = lib_math.nms(l, t, r, b, score, nms_threshold)
            if DEBUG_YOLO: print(f"Indices kept after NMS ({len(keep_indices)}): {keep_indices}")
            l, t, r, b = l[keep_indices], t[keep_indices], r[keep_indices], b[keep_indices]
            # score_nms = score[keep_indices] # Если нужны score после NMS

            if DEBUG_YOLO and len(l) > 0:
                 print(f"Boxes after NMS (first 5):")
                 for k in range(min(5, len(l))):
                      print(f"  Box {k}: l={l[k]:.2f}, t={t[k]:.2f}, r={r[k]:.2f}, b={b[k]:.2f}")

            # ... (Scaling, Clipping, Filtering остаются без изменений) ...
            if DEBUG_YOLO: print(f"Scaling boxes back to original size ({W_orig}x{H_orig}) using scale={img_scale:.4f}")
            if abs(img_scale - 1.0) > 1e-6:
                padded_w = target_width; padded_h = target_height; offset_x = (padded_w - W_orig * img_scale) / 2; offset_y = (padded_h - H_orig * img_scale) / 2
                if DEBUG_YOLO: print(f"  Padded W={padded_w}, H={padded_h}. Offset X={offset_x:.2f}, Y={offset_y:.2f}")
                l_before, t_before, r_before, b_before = l.copy(), t.copy(), r.copy(), b.copy()
                l = (l - offset_x) / img_scale; t = (t - offset_y) / img_scale; r = (r - offset_x) / img_scale; b = (b - offset_y) / img_scale
                if DEBUG_YOLO and len(l) > 0:
                     print(f"Boxes after scaling (first 5):")
                     for k in range(min(5, len(l))): print(f"  Box {k}: l={l[k]:.2f} (was {l_before[k]:.2f}), t={t[k]:.2f} (was {t_before[k]:.2f}), r={r[k]:.2f} (was {r_before[k]:.2f}), b={b[k]:.2f} (was {b_before[k]:.2f})")
            else:
                if DEBUG_YOLO: print("  No scaling needed (img_scale is ~1.0).")

            if DEBUG_YOLO: print(f"Clipping boxes to 0..{W_orig} and 0..{H_orig}")
            l_clipped = np.clip(l, 0, W_orig); t_clipped = np.clip(t, 0, H_orig); r_clipped = np.clip(r, 0, W_orig); b_clipped = np.clip(b, 0, H_orig)

            if DEBUG_YOLO: print(f"Filtering by min_face_size: {min_face_size}")
            faces = []
            for idx, (l_i, t_i, r_i, b_i) in enumerate(zip(l_clipped, t_clipped, r_clipped, b_clipped)):
                width = r_i - l_i; height = b_i - t_i
                if width >= min_face_size and height >= min_face_size:
                    final_face = (float(l_i), float(t_i), float(r_i), float(b_i))
                    faces.append(final_face)
                    if DEBUG_YOLO and len(faces) <= 5: print(f"  Face {len(faces)-1} ADDED: w={width:.1f}, h={height:.1f} -> {final_face}")
                elif DEBUG_YOLO and len(l_clipped) < 10: print(f"  Face {idx} REJECTED: w={width:.1f}, h={height:.1f} (min size: {min_face_size})")

            faces_per_batch.append(faces)
            if DEBUG_YOLO: print(f"--- Image {i} processing finished. Found {len(faces)} faces. ---")

        return faces_per_batch
