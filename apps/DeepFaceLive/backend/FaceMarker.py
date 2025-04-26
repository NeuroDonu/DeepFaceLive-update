# C:\Users\neurodonu\Downloads\DeepFaceLive\DeepFaceLive\apps\DeepFaceLive\backend\FaceMarker.py
# Код без отладочных принтов и сохранения изображений

import time
from enum import IntEnum
import numpy as np
from modelhub import onnx as onnx_models
from modelhub import cv as cv_models

from xlib import os as lib_os
from xlib.face import ELandmarks2D, FLandmarks2D, FPose
from xlib.image import ImageProcessor
from xlib.mp import csw as lib_csw

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendSignal, BackendWeakHeap, BackendWorker,
                          BackendWorkerState)


class MarkerType(IntEnum):
    OPENCV_LBF = 0
    GOOGLE_FACEMESH = 1
    INSIGHT_2D106 = 2

MarkerTypeNames = ['OpenCV LBF','Google FaceMesh','InsightFace_2D106']

class FaceMarker(BackendHost):
    def __init__(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, backend_db : BackendDB = None):
        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=FaceMarkerWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out, ] )

    def get_control_sheet(self) -> 'Sheet.Host': return super().get_control_sheet()


class FaceMarkerWorker(BackendWorker):
    def get_state(self) -> 'WorkerState': return super().get_state()
    def get_control_sheet(self) -> 'Sheet.Worker': return super().get_control_sheet()

    def on_start(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal,
                       bc_in : BackendConnection,
                       bc_out : BackendConnection,
                       ):
        self.weak_heap = weak_heap
        self.reemit_frame_signal = reemit_frame_signal
        self.bc_in = bc_in
        self.bc_out = bc_out
        self.pending_bcd = None
        self.opencv_lbf = None
        self.google_facemesh = None
        self.insightface_2d106 = None
        self.temporal_lmrks = []

        lib_os.set_timer_resolution(1)

        state, cs = self.get_state(), self.get_control_sheet()
        cs.marker_type.call_on_selected(self.on_cs_marker_type)
        cs.device.call_on_selected(self.on_cs_devices)
        cs.marker_coverage.call_on_number(self.on_cs_marker_coverage)
        cs.temporal_smoothing.call_on_number(self.on_cs_temporal_smoothing)

        cs.marker_type.enable()
        cs.marker_type.set_choices(MarkerType, MarkerTypeNames, none_choice_name=None)
        cs.marker_type.select(state.marker_type if state.marker_type is not None else MarkerType.GOOGLE_FACEMESH)

    def on_cs_marker_type(self, idx, marker_type):
        state, cs = self.get_state(), self.get_control_sheet()

        if state.marker_type == marker_type:
            cs.device.enable()
            if marker_type == MarkerType.OPENCV_LBF:
                cs.device.set_choices(['CPU'], none_choice_name='@misc.menu_select')
                cs.device.select(state.opencv_lbf_state.device)
            elif marker_type == MarkerType.GOOGLE_FACEMESH:
                cs.device.set_choices(onnx_models.FaceMesh.get_available_devices(), none_choice_name='@misc.menu_select')
                cs.device.select(state.google_facemesh_state.device)
            elif marker_type == MarkerType.INSIGHT_2D106:
                cs.device.set_choices(onnx_models.InsightFace2D106.get_available_devices(), none_choice_name='@misc.menu_select')
                cs.device.select(state.insightface_2d106_state.device)

        else:
            state.marker_type = marker_type
            self.save_state()
            self.restart()

    def on_cs_devices(self, idx, device):
        state, cs = self.get_state(), self.get_control_sheet()
        marker_type = state.marker_type

        if device is not None and \
            ( (marker_type == MarkerType.OPENCV_LBF and state.opencv_lbf_state.device == device) or \
              (marker_type == MarkerType.GOOGLE_FACEMESH and state.google_facemesh_state.device == device) or \
              (marker_type == MarkerType.INSIGHT_2D106 and state.insightface_2d106_state.device == device) ):
            marker_state = state.get_marker_state()

            # Выносим загрузку моделей сюда, чтобы она происходила только при успешном выборе устройства
            try:
                if state.marker_type == MarkerType.OPENCV_LBF:
                    self.opencv_lbf = cv_models.FaceMarkerLBF()
                elif state.marker_type == MarkerType.GOOGLE_FACEMESH:
                    self.google_facemesh = onnx_models.FaceMesh(state.google_facemesh_state.device)
                elif state.marker_type == MarkerType.INSIGHT_2D106:
                    self.insightface_2d106 = onnx_models.InsightFace2D106(state.insightface_2d106_state.device)
            except Exception as e:
                 print(f"!!! Error loading marker model for {MarkerTypeNames[marker_type]} on device {device}: {e}")
                 self.opencv_lbf = self.google_facemesh = self.insightface_2d106 = None
                 return # Прерываем настройку, т.к. модель не загрузилась

            # Настройка UI после успешной загрузки модели
            cs.marker_coverage.enable()
            cs.marker_coverage.set_config(lib_csw.Number.Config(min=0.1, max=3.0, step=0.1, decimals=1, allow_instant_update=True))

            marker_coverage = marker_state.marker_coverage
            default_coverage = {MarkerType.OPENCV_LBF: 1.1, MarkerType.GOOGLE_FACEMESH: 1.4, MarkerType.INSIGHT_2D106: 1.6}.get(marker_type, 1.4)
            cs.marker_coverage.set_number(marker_coverage if marker_coverage is not None else default_coverage)

            cs.temporal_smoothing.enable()
            cs.temporal_smoothing.set_config(lib_csw.Number.Config(min=1, max=150, step=1, allow_instant_update=True))
            cs.temporal_smoothing.set_number(marker_state.temporal_smoothing if marker_state.temporal_smoothing is not None else 1)

        else: # Если выбрано None или другое устройство
            if marker_type == MarkerType.OPENCV_LBF:
                state.opencv_lbf_state.device = device
            elif marker_type == MarkerType.GOOGLE_FACEMESH:
                state.google_facemesh_state.device = device
            elif marker_type == MarkerType.INSIGHT_2D106:
                state.insightface_2d106_state.device = device
            self.save_state()
            self.restart()


    def on_cs_marker_coverage(self, marker_coverage):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.marker_coverage.get_config()
        marker_coverage = state.get_marker_state().marker_coverage = np.clip(marker_coverage, cfg.min, cfg.max)
        cs.marker_coverage.set_number(marker_coverage)
        self.save_state()
        self.reemit_frame_signal.send()


    def on_cs_temporal_smoothing(self, temporal_smoothing):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.temporal_smoothing.get_config()
        temporal_smoothing = state.get_marker_state().temporal_smoothing = int(np.clip(temporal_smoothing,  cfg.min, cfg.max))
        if temporal_smoothing == 1:
            self.temporal_lmrks = [] # Сбрасываем буфер при отключении сглаживания
        cs.temporal_smoothing.set_number(temporal_smoothing)
        self.save_state()
        self.reemit_frame_signal.send()


    def on_tick(self):
        state, cs = self.get_state(), self.get_control_sheet()

        if self.pending_bcd is None:
            self.start_profile_timing()

            bcd = self.bc_in.read(timeout=0.005)
            if bcd is not None:
                bcd.assign_weak_heap(self.weak_heap)
                is_frame_reemitted = bcd.get_is_frame_reemitted()

                marker_type = state.marker_type
                marker_state = state.get_marker_state() # Получаем актуальные настройки покрытия и сглаживания

                # Проверяем, загружена ли нужная модель
                is_opencv_lbf = marker_type == MarkerType.OPENCV_LBF and self.opencv_lbf is not None
                is_google_facemesh = marker_type == MarkerType.GOOGLE_FACEMESH and self.google_facemesh is not None
                is_insightface_2d106 = marker_type == MarkerType.INSIGHT_2D106 and self.insightface_2d106 is not None
                is_marker_loaded = is_opencv_lbf or is_google_facemesh or is_insightface_2d106

                if marker_type is not None and is_marker_loaded:
                    frame_image = bcd.get_image(bcd.get_frame_image_name())

                    if frame_image is not None:
                        fsi_list = bcd.get_face_swap_info_list()

                        # Синхронизируем размер буфера сглаживания с количеством лиц
                        if marker_state.temporal_smoothing is not None and marker_state.temporal_smoothing != 1 and len(self.temporal_lmrks) != len(fsi_list):
                           # Аккуратно изменяем размер, сохраняя существующие данные, если возможно
                           new_temporal_lmrks = [[] for _ in range(len(fsi_list))]
                           for i in range(min(len(self.temporal_lmrks), len(fsi_list))):
                               new_temporal_lmrks[i] = self.temporal_lmrks[i]
                           self.temporal_lmrks = new_temporal_lmrks


                        for face_id, fsi in enumerate(fsi_list):
                            if fsi.face_urect is not None:
                                # Определяем целевой размер для маркера
                                target_size = 0
                                if is_opencv_lbf: target_size = 256
                                elif is_google_facemesh: target_size = 192
                                elif is_insightface_2d106: target_size = 192
                                else:
                                    continue # Пропускаем это лицо

                                # Вырезаем лицо
                                face_image = None
                                face_uni_mat = None
                                try:
                                    # Используем актуальное значение marker_coverage из marker_state
                                    current_coverage = marker_state.marker_coverage
                                    if current_coverage is None: # Fallback на дефолтное, если еще не установлено
                                        current_coverage = {MarkerType.OPENCV_LBF: 1.1, MarkerType.GOOGLE_FACEMESH: 1.4, MarkerType.INSIGHT_2D106: 1.6}.get(marker_type, 1.4)

                                    face_image, face_uni_mat = fsi.face_urect.cut(frame_image, current_coverage, target_size)

                                except Exception as e:
                                     # В рабочей версии можно логировать ошибку, если нужно
                                     # print(f"Error during face cut: {e}")
                                     continue # Пропускаем это лицо, если не удалось вырезать

                                # Получаем размеры вырезанного лица
                                if face_image is None: continue
                                _,H,W,_ = ImageProcessor(face_image).get_dims()
                                lmrks = None # Инициализируем

                                # Запускаем маркер
                                try:
                                    if is_opencv_lbf: lmrks = self.opencv_lbf.extract(face_image)[0]
                                    elif is_google_facemesh: lmrks = self.google_facemesh.extract(face_image)[0]
                                    elif is_insightface_2d106: lmrks = self.insightface_2d106.extract(face_image)[0]
                                except Exception as e:
                                     # В рабочей версии можно логировать ошибку
                                     # print(f"Error during marker extraction: {e}")
                                     pass # lmrks останется None

                                # Обработка лендмарков
                                if lmrks is not None and lmrks.size > 0:
                                    current_smoothing_len = marker_state.temporal_smoothing
                                    if current_smoothing_len is None: current_smoothing_len = 1

                                    if current_smoothing_len > 1:
                                        if face_id < len(self.temporal_lmrks):
                                            if not is_frame_reemitted or len(self.temporal_lmrks[face_id]) == 0:
                                                self.temporal_lmrks[face_id].append(lmrks)
                                            self.temporal_lmrks[face_id] = self.temporal_lmrks[face_id][-current_smoothing_len:]
                                            lmrks_processed = np.mean(self.temporal_lmrks[face_id], 0 )
                                        else:
                                            lmrks_processed = lmrks # Используем сырые, если face_id за пределами буфера
                                    else:
                                        lmrks_processed = lmrks

                                    # Расчет позы для FaceMesh
                                    if is_google_facemesh:
                                        try:
                                            fsi.face_pose = FPose.from_3D_468_landmarks(lmrks_processed)
                                        except Exception as e:
                                            # print(f"Error calculating pose: {e}")
                                            fsi.face_pose = None

                                    # Нормализация координат
                                    lmrks_normalized = None
                                    landmark_type = None
                                    if is_opencv_lbf:
                                        lmrks_normalized = lmrks_processed / (W,H); landmark_type = ELandmarks2D.L68
                                    elif is_google_facemesh:
                                        if lmrks_processed.shape[-1] >= 2:
                                            lmrks_normalized = lmrks_processed[...,0:2] / (W,H); landmark_type = ELandmarks2D.L468
                                    elif is_insightface_2d106:
                                         if lmrks_processed.shape[-1] >= 2:
                                            lmrks_normalized = lmrks_processed[...,0:2] / (W,H); landmark_type = ELandmarks2D.L106

                                    # Создание объекта FLandmarks2D и трансформация обратно
                                    if lmrks_normalized is not None and landmark_type is not None and face_uni_mat is not None:
                                        try:
                                            face_ulmrks = FLandmarks2D.create (landmark_type, lmrks_normalized)
                                            face_ulmrks = face_ulmrks.transform(face_uni_mat, invert=True)
                                            fsi.face_ulmrks = face_ulmrks
                                        except Exception as e:
                                             # print(f"Error creating/transforming landmarks: {e}")
                                             fsi.face_ulmrks = None
                                    else:
                                        fsi.face_ulmrks = None

                                else: # Если маркер не вернул лендмарки
                                    fsi.face_ulmrks = None
                                    fsi.face_pose = None
                            else: # Если fsi.face_urect is None
                                 pass # Пропускаем лицо без рамки

                    self.stop_profile_timing()
                else:
                    pass

                if bcd is not None:
                     self.pending_bcd = bcd

            if self.pending_bcd is not None:
                if self.bc_out.is_full_read(1):
                    self.bc_out.write(self.pending_bcd)
                    self.pending_bcd = None
                else:
                    time.sleep(0.001)


# --- Остальные классы без изменений ---
class MarkerState(BackendWorkerState):
    marker_coverage : float = None
    temporal_smoothing : int = None

class OpenCVLBFState(BackendWorkerState):
    device = None

class GoogleFaceMeshState(BackendWorkerState):
    device = None

class Insight2D106State(BackendWorkerState):
    device = None

class WorkerState(BackendWorkerState):
    def __init__(self):
        self.marker_type : MarkerType = None
        self.marker_state = {}
        self.opencv_lbf_state = OpenCVLBFState()
        self.google_facemesh_state = GoogleFaceMeshState()
        self.insightface_2d106_state = Insight2D106State()

    def get_marker_state(self) -> MarkerState:
        state = self.marker_state.setdefault(self.marker_type, MarkerState())
        return state

class Sheet:
    class Host(lib_csw.Sheet.Host):
        def __init__(self):
            super().__init__()
            self.marker_type = lib_csw.DynamicSingleSwitch.Client()
            self.device = lib_csw.DynamicSingleSwitch.Client()
            self.marker_coverage = lib_csw.Number.Client()
            self.temporal_smoothing = lib_csw.Number.Client()

    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.marker_type = lib_csw.DynamicSingleSwitch.Host()
            self.device = lib_csw.DynamicSingleSwitch.Host()
            self.marker_coverage = lib_csw.Number.Host()
            self.temporal_smoothing = lib_csw.Number.Host()