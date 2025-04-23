# !!! НАЧАЛО ПОЛНОГО КОДА ФАЙЛА FaceSwapInsight.py !!!
import time
from pathlib import Path
import cv2
import numpy as np
from modelhub.onnx import InsightFace2D106, InsightFaceSwap, YoloV8Face
from xlib import cv as lib_cv2
from xlib import os as lib_os
from xlib import path as lib_path
from xlib.face import ELandmarks2D, FLandmarks2D, FRect
from xlib.image.ImageProcessor import ImageProcessor
from xlib.mp import csw as lib_csw
import traceback # <<<--- ДОБАВЛЕНО ИСПРАВЛЕНИЕ 1: импорт traceback

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendSignal, BackendWeakHeap, BackendWorker,
                          BackendWorkerState)


class FaceSwapInsight(BackendHost):
    def __init__(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, faces_path : Path, backend_db : BackendDB = None,
                  id : int = 0):
        self._id = id
        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=FaceSwapInsightWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out, faces_path])

    def get_control_sheet(self) -> 'Sheet.Host': return super().get_control_sheet()

    def _get_name(self):
        return super()._get_name()

class FaceSwapInsightWorker(BackendWorker):
    def get_state(self) -> 'WorkerState': return super().get_state()
    def get_control_sheet(self) -> 'Sheet.Worker': return super().get_control_sheet()

    def on_start(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, faces_path : Path):
        self.weak_heap = weak_heap
        self.reemit_frame_signal = reemit_frame_signal
        self.bc_in = bc_in
        self.bc_out = bc_out
        self.faces_path = faces_path

        self.pending_bcd = None

        self.swap_model : InsightFaceSwap = None
        self.face_detector : YoloV8Face = None      # Добавим тип для ясности
        self.face_marker : InsightFace2D106 = None  # Добавим тип для ясности

        self.target_face_img = None
        self.face_vector = None

        lib_os.set_timer_resolution(1)

        state, cs = self.get_state(), self.get_control_sheet()

        cs.device.call_on_selected(self.on_cs_device)
        cs.face.call_on_selected(self.on_cs_face)
        cs.adjust_c.call_on_number(self.on_cs_adjust_c)
        cs.adjust_x.call_on_number(self.on_cs_adjust_x)
        cs.adjust_y.call_on_number(self.on_cs_adjust_y)

        cs.animator_face_id.call_on_number(self.on_cs_animator_face_id)
        cs.update_faces.call_on_signal(self.update_faces)

        cs.device.enable()
        cs.device.set_choices( InsightFaceSwap.get_available_devices(), none_choice_name='@misc.menu_select')
        cs.device.select(state.device)

    def update_faces(self):
        state, cs = self.get_state(), self.get_control_sheet()
        cs.face.set_choices([face_path.name for face_path in lib_path.get_files_paths(self.faces_path, extensions=['.jpg','.jpeg','.png'])], none_choice_name='@misc.menu_select')


    def on_cs_device(self, idx, device):
        state, cs = self.get_state(), self.get_control_sheet()
        # --- Немного изменим логику для корректной перезагрузки при смене устройства ---
        if device is not None:
            if state.device != device:
                #print(f"Insight Worker: Device changed from '{state.device}' to '{device}'. Restarting.")
                state.device = device
                self.face_vector = None # Сбросим вектор при смене устройства
                self.save_state()
                self.restart() # Перезапускаем для применения нового устройства
                return # Выходим, т.к. воркер перезапустится

            # Устройство выбрано и оно совпадает с текущим (или это первый запуск)
            if self.swap_model is None: # Инициализируем модели только если они еще не созданы
                #print(f"Insight Worker: Initializing models on device: {device}")
                try:
                    self.swap_model = InsightFaceSwap(device)
                    self.face_detector = YoloV8Face(device)
                    self.face_marker = InsightFace2D106(device)
                    #print("Insight Worker: Models initialized.")
                except Exception as e:
                    #print(f"Insight Worker: ***** EXCEPTION during model initialization *****")
                    traceback.print_exc()
                    # Отключить UI и показать ошибку?
                    cs.face.disable(); cs.adjust_c.disable(); cs.adjust_x.disable(); cs.adjust_y.disable(); cs.animator_face_id.disable(); cs.update_faces.disable()
                    self.swap_model = self.face_detector = self.face_marker = None
                    # Показать ошибку в UI?
                    return # Прекращаем настройку

            # Активируем остальные контролы
            #print("Insight Worker: Enabling UI controls.")
            cs.face.enable()
            self.update_faces()
            cs.face.select(state.face)

            cs.adjust_c.enable()
            cs.adjust_c.set_config(lib_csw.Number.Config(min=1.0, max=2.0, step=0.01, decimals=2, allow_instant_update=True))
            adjust_c = state.adjust_c if state.adjust_c is not None else 1.55
            cs.adjust_c.set_number(adjust_c)

            cs.adjust_x.enable()
            cs.adjust_x.set_config(lib_csw.Number.Config(min=-0.5, max=0.5, step=0.01, decimals=2, allow_instant_update=True))
            adjust_x = state.adjust_x if state.adjust_x is not None else 0.0
            cs.adjust_x.set_number(adjust_x)

            cs.adjust_y.enable()
            cs.adjust_y.set_config(lib_csw.Number.Config(min=-0.5, max=0.5, step=0.01, decimals=2, allow_instant_update=True))
            adjust_y = state.adjust_y if state.adjust_y is not None else -0.15
            cs.adjust_y.set_number(adjust_y)

            cs.animator_face_id.enable()
            cs.animator_face_id.set_config(lib_csw.Number.Config(min=0, max=16, step=1, decimals=0, allow_instant_update=True))
            cs.animator_face_id.set_number(state.animator_face_id if state.animator_face_id is not None else 0)

            cs.update_faces.enable()

        else: # device is None
            #print("Insight Worker: Device deselected.")
            if state.device is not None:
                 state.device = None
                 self.save_state()
            # Выгружаем модели
            self.swap_model = self.face_detector = self.face_marker = None
            self.face_vector = None
            # Отключаем контролы
            cs.face.disable(); cs.adjust_c.disable(); cs.adjust_x.disable(); cs.adjust_y.disable(); cs.animator_face_id.disable(); cs.update_faces.disable()


    def on_cs_face(self, idx, face):
        state, cs = self.get_state(), self.get_control_sheet()
        #print(f"Insight Worker: on_cs_face changed to: {face}") # Отладка

        state.face = face
        self.face_vector = None # Сбрасываем вектор при смене лица
        self.target_face_img = None

        if face is not None:
            face_filepath = self.faces_path / face
            #print(f"  Insight: Loading target image: {face_filepath}")
            try:
                self.target_face_img = lib_cv2.imread(face_filepath)
                if self.target_face_img is None:
                     #print(f"  Insight: ERROR - Failed to read image file: {face_filepath}")
                     raise ValueError("imread failed")
                #print(f"  Insight: Target image loaded. Shape: {self.target_face_img.shape}")

            except Exception as e:
                #print(f"  Insight: Exception loading image: {e}")
                traceback.print_exc()
                cs.face.unselect() # Сбрасываем выбор в UI

        self.save_state()
        self.reemit_frame_signal.send() # Сигнал для обновления, если нужно

    def on_cs_adjust_c(self, adjust_c):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.adjust_c.get_config()
        adjust_c = state.adjust_c = np.clip(adjust_c, cfg.min, cfg.max)
        cs.adjust_c.set_number(adjust_c)
        #print(f"Insight Worker: Adjust C set to: {adjust_c}. Resetting vector.") # Отладка
        self.face_vector = None # Сброс вектора при изменении параметра
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_adjust_x(self, adjust_x):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.adjust_x.get_config()
        adjust_x = state.adjust_x = np.clip(adjust_x, cfg.min, cfg.max)
        cs.adjust_x.set_number(adjust_x)
        #print(f"Insight Worker: Adjust X set to: {adjust_x}. Resetting vector.") # Отладка
        self.face_vector = None # Сброс вектора при изменении параметра
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_adjust_y(self, adjust_y):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.adjust_y.get_config()
        adjust_y = state.adjust_y = np.clip(adjust_y, cfg.min, cfg.max)
        cs.adjust_y.set_number(adjust_y)
        #print(f"Insight Worker: Adjust Y set to: {adjust_y}. Resetting vector.") # Отладка
        self.face_vector = None # Сброс вектора при изменении параметра
        self.save_state()
        self.reemit_frame_signal.send()


    def on_cs_animator_face_id(self, animator_face_id):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.animator_face_id.get_config()
        animator_face_id = state.animator_face_id = int(np.clip(animator_face_id, cfg.min, cfg.max))
        cs.animator_face_id.set_number(animator_face_id)
        #print(f"Insight Worker: animator_face_id set to: {animator_face_id}") # Отладка
        self.save_state()
        self.reemit_frame_signal.send()


    def on_tick(self):
        state, cs = self.get_state(), self.get_control_sheet()

        # --- Отладка получения вектора ---
        if self.face_vector is None and self.target_face_img is not None and self.face_detector is not None and self.face_marker is not None and self.swap_model is not None:
            #print("Insight Tick: Attempting to get face vector...")
            try:
                rects = self.face_detector.extract (self.target_face_img, threshold=0.5)[0]
                #print(f"  Insight Vector: Detected {len(rects)} rects in target image.")
                if len(rects) > 0:
                    _,H,W,_ = ImageProcessor(self.target_face_img).get_dims()
                    u_rects = [ FRect.from_ltrb( (l/W, t/H, r/W, b/H) ) for l,t,r,b in rects ]
                    face_urect = FRect.sort_by_area_size(u_rects)[0]
                    #print(f"    Insight Vector: Largest u_rect: {face_urect}")

                    face_image, face_uni_mat = face_urect.cut(self.target_face_img, 1.6, 192)
                    #print(f"    Insight Vector: Extracted face_image shape: {face_image.shape if face_image is not None else None}")

                    lmrks_result = self.face_marker.extract(face_image)
                    if lmrks_result is None or len(lmrks_result) == 0: raise ValueError("Face marker failed to extract landmarks.")
                    lmrks = lmrks_result[0]
                    #print(f"    Insight Vector: Extracted landmarks shape: {lmrks.shape if lmrks is not None else None}")
                    lmrks = lmrks[...,0:2] / (192,192)

                    face_ulmrks = FLandmarks2D.create (ELandmarks2D.L106, lmrks).transform(face_uni_mat, invert=True)
                    # --- ИСПРАВЛЕНО ЗДЕСЬ ---
                    # Проверяем, есть ли метод get_points, иначе пытаемся получить длину напрямую (менее вероятно)
                    lmrks_count = len(face_ulmrks.get_points()) if hasattr(face_ulmrks, 'get_points') else (len(face_ulmrks) if hasattr(face_ulmrks, '__len__') else 'N/A')
                    #print(f"    Insight Vector: Transformed face_ulmrks count: {lmrks_count}")
                    # ------------------------

                    vector_adjust_c = state.adjust_c if state.adjust_c is not None else 1.55
                    vector_adjust_x = state.adjust_x if state.adjust_x is not None else 0.0
                    vector_adjust_y = state.adjust_y if state.adjust_y is not None else -0.15

                    face_align_img, _ = face_ulmrks.cut(self.target_face_img, vector_adjust_c,
                                                            self.swap_model.get_face_vector_input_size(),
                                                            x_offset=vector_adjust_x,
                                                            y_offset=vector_adjust_y)
                    #print(f"    Insight Vector: Final aligned image shape for vector: {face_align_img.shape if face_align_img is not None else None}")

                    self.face_vector = self.swap_model.get_face_vector(face_align_img)
                    #print(f"    Insight Vector: Got face_vector! Shape: {self.face_vector.shape if self.face_vector is not None else None}")
                else:
                    print("    Insight Vector: No faces detected in target image!")

            except Exception as e:
                #print(f"  Insight Vector: ***** EXCEPTION during get_face_vector *****")
                traceback.print_exc()
                self.face_vector = None

            except Exception as e:
                #print(f"  Insight Vector: ***** EXCEPTION during get_face_vector *****")
                traceback.print_exc() # traceback уже импортирован
                self.face_vector = None # Явно ставим None при ошибке

        # --- Обработка кадра ---
        if self.pending_bcd is None:
            # --- УДАЛЕНО if not self.is_busy(): ---
            self.start_profile_timing()
            bcd = self.bc_in.read(timeout=0.005)
            if bcd is not None:
                bcd.assign_weak_heap(self.weak_heap)

                swap_model = self.swap_model
                #print(f"Insight Process Tick: Checking swap. swap_model is None? {swap_model is None}. face_vector is None? {self.face_vector is None}")

                if swap_model is not None and self.face_vector is not None:
                    # ... (остальной код обработки кадра без изменений) ...
                    #print("  Insight Process Tick: Model and vector OK. Entering face loop.")
                    try:
                        target_face_id = state.animator_face_id if state.animator_face_id is not None else 0
                        for i, fsi in enumerate(bcd.get_face_swap_info_list()):
                            #print(f"    Insight: Checking face idx {i}. Target ID? {target_face_id}")
                            if target_face_id == i:
                                # ... (остальной код внутри if target_face_id == i) ...
                                #print(f"      Insight: Processing face {i}...")
                                face_align_image = bcd.get_image(fsi.face_align_image_name)
                                if face_align_image is None: print(f"        Insight: ERROR! face_align_image is None for face {i}!"); continue
                                else: print(f"        Insight: Got face_align_image shape {face_align_image.shape}")
                                _,H,W,_ = ImageProcessor(face_align_image).get_dims()
                                #print(f"        Insight: Calling swap_model.generate...")
                                try: anim_image = swap_model.generate(face_align_image, self.face_vector)
                                except Exception as e_gen: print(f"        Insight: ***** EXCEPTION during swap_model.generate() *****"); traceback.print_exc(); anim_image = None
                                #print(f"        Insight: Generate result shape: {anim_image.shape if anim_image is not None else None}")
                                if anim_image is not None:
                                    try:
                                        anim_image = ImageProcessor(anim_image).resize((W,H)).get_image('HWC')
                                        #print(f"        Insight: Resized result shape: {anim_image.shape}")
                                        fsi.face_align_mask_name = f'{fsi.face_align_image_name}_mask'; fsi.face_swap_image_name = f'{fsi.face_align_image_name}_swapped'; fsi.face_swap_mask_name  = f'{fsi.face_swap_image_name}_mask'
                                        #print(f"        Insight: Setting images in BCD...")
                                        bcd.set_image(fsi.face_swap_image_name, anim_image)
                                        white_mask = np.full_like(anim_image, 255, dtype=np.uint8)
                                        bcd.set_image(fsi.face_align_mask_name, white_mask); bcd.set_image(fsi.face_swap_mask_name, white_mask)
                                        #print(f"        Insight: Images set for face {i}.")
                                    except Exception as e_postproc: print(f"        Insight: ***** EXCEPTION during result postprocessing/saving *****"); traceback.print_exc()
                                else: print(f"        Insight: ERROR - swap_model.generate returned None!")
                                break
                    except Exception as e_loop: print(f"    Insight: ***** UNCAUGHT EXCEPTION during face processing loop *****"); traceback.print_exc()
                else: print(f"  Insight Process Tick: Skipping swap. Reason: swap_model is None? {swap_model is None}, face_vector is None? {self.face_vector is None}")
                self.stop_profile_timing()
                self.pending_bcd = bcd
                # else: # Раскомментировать для отладки отсутствия BCD
                #    #print("Insight Process Tick: No input BCD.")

        # --- Отправка результата ---
        if self.pending_bcd is not None:
            if self.bc_out.is_full_read(1):
                self.bc_out.write(self.pending_bcd)
                self.pending_bcd = None
            else:
                time.sleep(0.001)

# --- Классы Sheet и WorkerState без изменений ---
class Sheet:
    class Host(lib_csw.Sheet.Host):
        def __init__(self):
            super().__init__()
            self.device = lib_csw.DynamicSingleSwitch.Client()
            self.face = lib_csw.DynamicSingleSwitch.Client()
            self.animator_face_id = lib_csw.Number.Client()
            self.update_faces = lib_csw.Signal.Client()
            self.adjust_c = lib_csw.Number.Client()
            self.adjust_x = lib_csw.Number.Client()
            self.adjust_y = lib_csw.Number.Client()


    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.device = lib_csw.DynamicSingleSwitch.Host()
            self.face = lib_csw.DynamicSingleSwitch.Host()
            self.animator_face_id = lib_csw.Number.Host()
            self.update_faces = lib_csw.Signal.Host()
            self.adjust_c = lib_csw.Number.Host()
            self.adjust_x = lib_csw.Number.Host()
            self.adjust_y = lib_csw.Number.Host()


class WorkerState(BackendWorkerState):
    device = None
    face : str = None
    animator_face_id : int = None
    adjust_c : float = None
    adjust_x : float = None
    adjust_y : float = None
# !!! КОНЕЦ ПОЛНОГО КОДА ФАЙЛА FaceSwapInsight.py !!!