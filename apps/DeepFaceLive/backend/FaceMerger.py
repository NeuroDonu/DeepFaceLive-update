# --- Полный код FaceMerger.py с исправлениями и форматированием ---
import time
import traceback # Убедимся, что импортирован

import cv2
import numexpr as ne
import numpy as np
from xlib import avecl as lib_cl # lib_cl должен содержать concat
from xlib import os as lib_os
from xlib.image import ImageProcessor
from xlib.mp import csw as lib_csw
from xlib.python import all_is_not_None

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendSignal, BackendWeakHeap, BackendWorker,
                          BackendWorkerState)


class FaceMerger(BackendHost):
    def __init__(self, weak_heap: BackendWeakHeap, reemit_frame_signal: BackendSignal, bc_in: BackendConnection, bc_out: BackendConnection, backend_db: BackendDB = None):
        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=FaceMergerWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out])

    def get_control_sheet(self) -> 'Sheet.Host':
        return super().get_control_sheet()


class FaceMergerWorker(BackendWorker):
    def get_state(self) -> 'WorkerState':
        return super().get_state()

    def get_control_sheet(self) -> 'Sheet.Worker':
        return super().get_control_sheet()

    def on_start(self, weak_heap: BackendWeakHeap, reemit_frame_signal: BackendSignal, bc_in: BackendConnection, bc_out: BackendConnection):
        self.weak_heap = weak_heap
        self.reemit_frame_signal = reemit_frame_signal
        self.bc_in = bc_in
        self.bc_out = bc_out
        self.pending_bcd = None

        lib_os.set_timer_resolution(1)

        state, cs = self.get_state(), self.get_control_sheet()
        cs.device.call_on_selected(self.on_cs_device)
        cs.face_x_offset.call_on_number(self.on_cs_face_x_offset)
        cs.face_y_offset.call_on_number(self.on_cs_face_y_offset)
        cs.face_scale.call_on_number(self.on_cs_face_scale)
        cs.face_mask_source.call_on_flag(self.on_cs_face_mask_source)
        cs.face_mask_celeb.call_on_flag(self.on_cs_face_mask_celeb)
        cs.face_mask_lmrks.call_on_flag(self.on_cs_face_mask_lmrks)
        cs.face_mask_x_offset.call_on_number(self.on_cs_face_mask_x_offset)
        cs.face_mask_y_offset.call_on_number(self.on_cs_face_mask_y_offset)
        cs.face_mask_scale.call_on_number(self.on_cs_face_mask_scale)
        cs.face_mask_erode.call_on_number(self.on_cs_mask_erode)
        cs.face_mask_blur.call_on_number(self.on_cs_mask_blur)
        cs.color_transfer.call_on_selected(self.on_cs_color_transfer)
        cs.interpolation.call_on_selected(self.on_cs_interpolation)
        cs.color_compression.call_on_number(self.on_cs_color_compression)
        cs.face_opacity.call_on_number(self.on_cs_face_opacity)

        cs.device.enable()
        cs.device.set_choices(['CPU'] + lib_cl.get_available_devices_info(), none_choice_name='@misc.menu_select')
        cs.device.select(state.device if state.device is not None else 'CPU')

    def on_cs_device(self, idxs, device):
        state, cs = self.get_state(), self.get_control_sheet()

        current_device_str = 'CPU'
        if isinstance(state.device, lib_cl.DeviceInfo):
            current_device_str = str(state.device)
        elif state.device == 'CPU':
            current_device_str = 'CPU'

        new_device_str = 'CPU'
        if isinstance(device, lib_cl.DeviceInfo):
            new_device_str = str(device)
        elif device == 'CPU':
            new_device_str = 'CPU'
        elif device is None or device == '@misc.menu_select':
            cs.face_x_offset.disable()
            cs.face_y_offset.disable()
            cs.face_scale.disable()
            cs.face_mask_source.disable()
            cs.face_mask_celeb.disable()
            cs.face_mask_lmrks.disable()
            cs.face_mask_x_offset.disable()
            cs.face_mask_y_offset.disable()
            cs.face_mask_scale.disable()
            cs.face_mask_erode.disable()
            cs.face_mask_blur.disable()
            cs.color_transfer.disable()
            cs.interpolation.disable()
            cs.color_compression.disable()
            cs.face_opacity.disable()
            return

        if new_device_str != current_device_str:
            state.device = device
            self.save_state()
            self.restart()
            return

        if isinstance(device, lib_cl.DeviceInfo):
            try:
                dev = lib_cl.get_device(device)
                lib_cl.set_default_device(dev)
            except Exception as e:
                #print(f"Ошибка настройки OpenCL устройства {device}: {e}")
                state.device = 'CPU'
                self.save_state()
                self.restart()
                return
        elif device == 'CPU':
            pass

        cs.face_x_offset.enable()
        cs.face_x_offset.set_config(lib_csw.Number.Config(min=-0.5, max=0.5, step=0.001, decimals=3, allow_instant_update=True))
        cs.face_x_offset.set_number(state.face_x_offset if state.face_x_offset is not None else 0.0)

        cs.face_y_offset.enable()
        cs.face_y_offset.set_config(lib_csw.Number.Config(min=-0.5, max=0.5, step=0.001, decimals=3, allow_instant_update=True))
        cs.face_y_offset.set_number(state.face_y_offset if state.face_y_offset is not None else 0.0)

        cs.face_scale.enable()
        cs.face_scale.set_config(lib_csw.Number.Config(min=0.5, max=1.5, step=0.01, decimals=2, allow_instant_update=True))
        cs.face_scale.set_number(state.face_scale if state.face_scale is not None else 1.0)

        cs.face_mask_source.enable()
        cs.face_mask_source.set_flag(state.face_mask_source if state.face_mask_source is not None else True)

        cs.face_mask_celeb.enable()
        cs.face_mask_celeb.set_flag(state.face_mask_celeb if state.face_mask_celeb is not None else True)

        cs.face_mask_lmrks.enable()
        cs.face_mask_lmrks.set_flag(state.face_mask_lmrks if state.face_mask_lmrks is not None else False)

        cs.face_mask_x_offset.enable()
        cs.face_mask_x_offset.set_config(lib_csw.Number.Config(min=-0.5, max=0.5, step=0.01, decimals=2, allow_instant_update=True))
        cs.face_mask_x_offset.set_number(state.face_mask_x_offset if state.face_mask_x_offset is not None else 0.0)

        cs.face_mask_y_offset.enable()
        cs.face_mask_y_offset.set_config(lib_csw.Number.Config(min=-0.5, max=0.5, step=0.01, decimals=2, allow_instant_update=True))
        cs.face_mask_y_offset.set_number(state.face_mask_y_offset if state.face_mask_y_offset is not None else 0.0)

        cs.face_mask_scale.enable()
        cs.face_mask_scale.set_config(lib_csw.Number.Config(min=0.1, max=10.0, step=0.1, decimals=1, allow_instant_update=True))
        cs.face_mask_scale.set_number(state.face_mask_scale if state.face_mask_scale is not None else 1.0)

        cs.face_mask_erode.enable()
        cs.face_mask_erode.set_config(lib_csw.Number.Config(min=-400, max=400, step=1, decimals=0, allow_instant_update=True))
        cs.face_mask_erode.set_number(state.face_mask_erode if state.face_mask_erode is not None else 5)

        cs.face_mask_blur.enable()
        cs.face_mask_blur.set_config(lib_csw.Number.Config(min=0, max=400, step=1, decimals=0, allow_instant_update=True))
        cs.face_mask_blur.set_number(state.face_mask_blur if state.face_mask_blur is not None else 25)

        cs.color_transfer.enable()
        cs.color_transfer.set_choices(['none', 'rct'])
        cs.color_transfer.select(state.color_transfer if state.color_transfer is not None else 'rct')

        cs.interpolation.enable()
        cs.interpolation.set_choices(['bilinear', 'bicubic', 'lanczos4'], none_choice_name=None)
        cs.interpolation.select(state.interpolation if state.interpolation is not None else 'bilinear')

        cs.color_compression.enable()
        cs.color_compression.set_config(lib_csw.Number.Config(min=0.0, max=127.0, step=0.1, decimals=1, allow_instant_update=True))
        cs.color_compression.set_number(state.color_compression if state.color_compression is not None else 1.0)

        cs.face_opacity.enable()
        cs.face_opacity.set_config(lib_csw.Number.Config(min=0.0, max=1.0, step=0.01, decimals=2, allow_instant_update=True))
        cs.face_opacity.set_number(state.face_opacity if state.face_opacity is not None else 1.0)

    # Обработчики изменений параметров
    def on_cs_face_x_offset(self, face_x_offset):
        state = self.get_state()
        cfg = self.get_control_sheet().face_x_offset.get_config()
        state.face_x_offset = float(np.clip(face_x_offset, cfg.min, cfg.max))
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_face_y_offset(self, face_y_offset):
        state = self.get_state()
        cfg = self.get_control_sheet().face_y_offset.get_config()
        state.face_y_offset = float(np.clip(face_y_offset, cfg.min, cfg.max))
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_face_scale(self, face_scale):
        state = self.get_state()
        cfg = self.get_control_sheet().face_scale.get_config()
        state.face_scale = float(np.clip(face_scale, cfg.min, cfg.max))
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_face_mask_source(self, face_mask_source):
        state = self.get_state()
        state.face_mask_source = face_mask_source
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_face_mask_celeb(self, face_mask_celeb):
        state = self.get_state()
        state.face_mask_celeb = face_mask_celeb
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_face_mask_lmrks(self, face_mask_lmrks):
        state = self.get_state()
        state.face_mask_lmrks = face_mask_lmrks
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_face_mask_x_offset(self, face_mask_x_offset):
        state = self.get_state()
        cfg = self.get_control_sheet().face_mask_x_offset.get_config()
        state.face_mask_x_offset = float(np.clip(face_mask_x_offset, cfg.min, cfg.max))
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_face_mask_y_offset(self, face_mask_y_offset):
        state = self.get_state()
        cfg = self.get_control_sheet().face_mask_y_offset.get_config()
        state.face_mask_y_offset = float(np.clip(face_mask_y_offset, cfg.min, cfg.max))
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_face_mask_scale(self, face_mask_scale):
        state = self.get_state()
        cfg = self.get_control_sheet().face_mask_scale.get_config()
        state.face_mask_scale = float(np.clip(face_mask_scale, cfg.min, cfg.max))
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_mask_erode(self, face_mask_erode):
        state = self.get_state()
        cfg = self.get_control_sheet().face_mask_erode.get_config()
        state.face_mask_erode = int(np.clip(face_mask_erode, cfg.min, cfg.max))
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_mask_blur(self, face_mask_blur):
        state = self.get_state()
        cfg = self.get_control_sheet().face_mask_blur.get_config()
        state.face_mask_blur = int(np.clip(face_mask_blur, cfg.min, cfg.max))
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_color_transfer(self, idx, color_transfer):
        state = self.get_state()
        state.color_transfer = color_transfer
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_interpolation(self, idx, interpolation):
        state = self.get_state()
        state.interpolation = interpolation
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_color_compression(self, color_compression):
        state = self.get_state()
        cfg = self.get_control_sheet().color_compression.get_config()
        state.color_compression = float(np.clip(color_compression, cfg.min, cfg.max))
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_face_opacity(self, face_opacity):
        state = self.get_state()
        cfg = self.get_control_sheet().face_opacity.get_config()
        state.face_opacity = float(np.clip(face_opacity, cfg.min, cfg.max))
        self.save_state()
        self.reemit_frame_signal.send()

    # Функция слияния на CPU
    _cpu_interp = {'bilinear': cv2.INTER_LINEAR, 'bicubic': cv2.INTER_CUBIC, 'lanczos4': cv2.INTER_LANCZOS4}

    def _merge_on_cpu(self, current_frame_image_float, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression):
        state = self.get_state()
        interpolation = self._cpu_interp.get(state.interpolation, cv2.INTER_LINEAR)

        boolean_masks = []
        if state.face_mask_source and face_align_mask_img is not None:
            mask_arr = ImageProcessor(face_align_mask_img).ch(1).to_ufloat32().get_image('HW')
            boolean_masks.append(mask_arr > 0.5)
        if state.face_mask_celeb and face_swap_mask_img is not None:
            mask_arr = ImageProcessor(face_swap_mask_img).ch(1).to_ufloat32().get_image('HW')
            boolean_masks.append(mask_arr > 0.5)
        if state.face_mask_lmrks and face_align_lmrks_mask_img is not None:
            mask_arr = ImageProcessor(face_align_lmrks_mask_img).ch(1).to_ufloat32().get_image('HW')
            boolean_masks.append(mask_arr > 0.5)

        if not boolean_masks:
            face_mask_ip = ImageProcessor(np.ones((face_resolution, face_resolution), dtype=np.float32))
        else:
            final_mask_bool = boolean_masks[0]
            for i in range(1, len(boolean_masks)):
                if final_mask_bool.shape == boolean_masks[i].shape:
                    np.logical_and(final_mask_bool, boolean_masks[i], out=final_mask_bool)
                else:
                    pass
                    #print(f"Предупреждение: Несоответствие размеров масок на CPU: {final_mask_bool.shape} vs {boolean_masks[i].shape}")
            face_mask_ip = ImageProcessor(final_mask_bool.astype(np.float32))

        erode_val = int(state.face_mask_erode)
        blur_val = int(state.face_mask_blur)
        face_mask_ip = face_mask_ip.erode_blur(erode_val, blur_val, fade_to_border=True)
        frame_face_mask = cv2.warpAffine(face_mask_ip.get_image('HWC'), aligned_to_source_uni_mat, (frame_width, frame_height), flags=interpolation | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        np.clip(frame_face_mask, 0.0, 1.0, out=frame_face_mask)
        if frame_face_mask.ndim == 2:
            frame_face_mask = np.expand_dims(frame_face_mask, axis=-1)

        face_swap_ip = ImageProcessor(face_swap_img).to_ufloat32()
        if state.color_transfer == 'rct' and face_align_img is not None:
            face_align_img_float = ImageProcessor(face_align_img).to_ufloat32().get_image('HWC')
            face_mask_float_hwc = face_mask_ip.get_image('HWC')
            if face_mask_float_hwc.ndim == 2: face_mask_float_hwc = np.expand_dims(face_mask_float_hwc, -1)
            try:
                 face_swap_ip = face_swap_ip.rct(like=face_align_img_float, mask=face_mask_float_hwc, like_mask=face_mask_float_hwc)
            except Exception as e:
                pass
                 #print(f"Предупреждение: RCT на CPU не удалось. Ошибка: {e}")


        frame_face_swap_img = cv2.warpAffine(face_swap_ip.get_image('HWC'), aligned_to_source_uni_mat, (frame_width, frame_height), flags=interpolation | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        np.clip(frame_face_swap_img, 0.0, 1.0, out=frame_face_swap_img)

        opacity = np.float32(state.face_opacity)
        one_f = np.float32(1.0)
        if opacity == 1.0:
            out_frame = ne.evaluate('current_frame_image_float * (one_f - frame_face_mask) + frame_face_swap_img * frame_face_mask')
        else:
            out_frame = ne.evaluate('current_frame_image_float * (one_f - frame_face_mask) + current_frame_image_float * frame_face_mask * (one_f - opacity) + frame_face_swap_img * frame_face_mask * opacity')

        if do_color_compression and state.color_compression > 0:
            color_compression_divisor = max(1.0, (128.0 - state.color_compression))
            out_frame = ne.evaluate('(floor(out_frame * color_compression_divisor) / color_compression_divisor) + (2.0 / color_compression_divisor)')
            np.clip(out_frame, 0.0, 1.0, out=out_frame)
        return out_frame

    # Функция слияния на GPU
    _gpu_interp = {'bilinear': lib_cl.EInterpolation.LINEAR, 'bicubic': lib_cl.EInterpolation.CUBIC, 'lanczos4': lib_cl.EInterpolation.LANCZOS4}

    def _merge_on_gpu(self, current_frame_image_t, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression):
        state = self.get_state()
        interpolation = self._gpu_interp.get(state.interpolation, lib_cl.EInterpolation.LINEAR)
        #print("DEBUG: Entering _merge_on_gpu") # Debug #print

        try: # --- Блок 1: Подготовка масок ---
            #print("DEBUG: Preparing mask tensors")
            mask_tensors = []

            def _add_mask_tensor(img):
                if img is not None:
                    #print(f"DEBUG _add_mask_tensor: Input img shape: {img.shape}, dtype: {img.dtype}")
                    if img.ndim == 2: img = np.expand_dims(img, axis=-1)
                    if img.ndim == 3 and img.shape[2] == 1:
                        #print(f"DEBUG _add_mask_tensor: Numpy mask shape before Tensor.from_value: {img.shape}")
                        tensor = lib_cl.Tensor.from_value(img)
                        #print(f"DEBUG _add_mask_tensor: Created tensor shape: {tensor.shape}") # Ожидаем (H, W, 1)
                        return tensor
                    elif img.ndim == 3 and img.shape[2] >= 3:
                        #print(f"Предупреждение: Маска имеет {img.shape[2]} каналов. Используется первый.")
                        img = img[..., 0:1]
                        #print(f"DEBUG _add_mask_tensor: Numpy mask shape before Tensor.from_value: {img.shape}")
                        tensor = lib_cl.Tensor.from_value(img)
                        #print(f"DEBUG _add_mask_tensor: Created tensor shape: {tensor.shape}") # Ожидаем (H, W, 1)
                        return tensor
                    else:
                        #print(f"Предупреждение: Неожиданное измерение маски: {img.shape}. Пропуск.")
                        return None
                return None

            if state.face_mask_source and face_align_mask_img is not None:
                tensor = _add_mask_tensor(face_align_mask_img)
                if tensor: mask_tensors.append(tensor); #print("DEBUG: Added source mask tensor")
            if state.face_mask_celeb and face_swap_mask_img is not None:
                tensor = _add_mask_tensor(face_swap_mask_img)
                if tensor: mask_tensors.append(tensor); #print("DEBUG: Added celeb mask tensor")
            if state.face_mask_lmrks and face_align_lmrks_mask_img is not None:
                tensor = _add_mask_tensor(face_align_lmrks_mask_img)
                if tensor: mask_tensors.append(tensor); #print("DEBUG: Added lmrks mask tensor")

            masks_count = len(mask_tensors)
            #print(f"DEBUG: Found {masks_count} mask tensors")
            if masks_count == 0:
                face_mask_t = lib_cl.Tensor(shape=(face_resolution, face_resolution, 1), dtype=np.float32, initializer=lib_cl.InitConst(1.0))
                #print(f"DEBUG: Created default ones mask tensor with shape: {face_mask_t.shape}")
            else:
                face_mask_t = mask_tensors[0]
                for i in range(1, masks_count):
                    #print(f"DEBUG: Combining mask {i}. Current shape: {face_mask_t.shape}, Next shape: {mask_tensors[i].shape}")
                    face_mask_t = lib_cl.any_wise("O = (I0 / 255.0f) * (I1 / 255.0f)", face_mask_t, mask_tensors[i], dtype=np.float32)
                #print("DEBUG: Applying threshold to combined mask")
                face_mask_t = lib_cl.any_wise("O = (I0 <= 0.5 ? 0.0f : 1.0f)", face_mask_t, dtype=np.float32)

            #print(f"DEBUG: Final combined mask shape before morph: {face_mask_t.shape}") # Ожидаем (H, W, 1)
            #print("DEBUG: Mask preparation done")
        except Exception as e:
            #print(f"!!!!!!!! ОШИБКА В БЛОКЕ 1 (Подготовка масок): {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        try: # --- Блок 2: Обработка маски (erode/blur) ---
            #print(f"DEBUG: Processing mask (erode/blur). Input shape: {face_mask_t.shape}") # Ожидаем (H, W, 1)
            erode_val = int(state.face_mask_erode)
            blur_val = int(state.face_mask_blur)
            face_mask_t = lib_cl.binary_morph(face_mask_t, erode_val, blur_val, fade_to_border=True, dtype=np.float32)
            #print(f"DEBUG: Mask shape after morph: {face_mask_t.shape}") # Ожидаем (H, W, 1)

            # --- Транспонируем маску в (C, H, W) перед ремаппингом ---
            if face_mask_t.shape[-1] == 1 and face_mask_t.ndim == 3:
                 mask_transpose_axes = (2, 0, 1)
                 #print(f"DEBUG: Transposing mask {face_mask_t.shape} using axes {mask_transpose_axes} before remap")
                 face_mask_t_chw = face_mask_t.transpose(mask_transpose_axes)
                 #print(f"DEBUG: Mask shape after transpose: {face_mask_t_chw.shape}") # Ожидаем (1, H, W)
            elif face_mask_t.ndim == 3 and face_mask_t.shape[0] == 1:
                 #print(f"DEBUG: Mask shape {face_mask_t.shape} is already (C, H, W), no transpose needed before remap.")
                 face_mask_t_chw = face_mask_t
            else:
                 #print(f"ПРЕДУПРЕЖДЕНИЕ: Неожиданная форма маски перед ремаппингом: {face_mask_t.shape}.")
                 face_mask_t_chw = face_mask_t

            # --- Явно вычисляем ожидаемую выходную форму ---
            expected_mask_output_shape_tuple = face_mask_t_chw.shape[:-2] + (frame_height, frame_width)
            #print(f"DEBUG: Calculated expected_mask_output_shape_tuple: {expected_mask_output_shape_tuple}") # Ожидаем (1, frame_H, frame_W)

            #print("DEBUG: Remapping mask")
            # Передаем ожидаемую форму в remap_np_affine
            frame_face_mask_t = lib_cl.remap_np_affine(face_mask_t_chw,
                                                       aligned_to_source_uni_mat,
                                                       interpolation=lib_cl.EInterpolation.LINEAR,
                                                       output_size=(frame_height, frame_width),
                                                       post_op_text='O = clamp(O, 0.0f, 1.0f);',
                                                       expected_o_shape_tuple=expected_mask_output_shape_tuple)

            #print(f"DEBUG: Mask shape after remap: {frame_face_mask_t.shape}") # Ожидаем (1, frame_H, frame_W)
            #print("DEBUG: Mask processing done")
        except Exception as e:
            #print(f"!!!!!!!! ОШИБКА В БЛОКЕ 2 (Обработка маски): {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        try: # --- Блок 3: Подготовка face_swap ---
            #print("DEBUG: Preparing face_swap tensor")
            face_swap_img_t = lib_cl.Tensor.from_value(face_swap_img).transpose((2, 0, 1), op_text='O = ((float)I) / 255.0f;', dtype=np.float32)
            #print(f"DEBUG: face_swap_img_t shape: {face_swap_img_t.shape}") # Ожидаем (3, H, W)

            if state.color_transfer == 'rct' and face_align_img is not None:
                #print("DEBUG: Applying RCT")
                face_align_img_t = lib_cl.Tensor.from_value(face_align_img).transpose((2, 0, 1), op_text='O = ((float)I) / 255.0f;', dtype=np.float32)
                #print(f"DEBUG: face_align_img_t shape: {face_align_img_t.shape}") # Ожидаем (3, H, W)
                # Используем face_mask_t формы (H, W, 1) для RCT
                #print(f"DEBUG: RCT Inputs - source_t: {face_swap_img_t.shape}, target_t: {face_align_img_t.shape}, target_mask_t: {face_mask_t.shape}, source_mask_t: {face_mask_t.shape}")
                face_swap_img_t = lib_cl.rct(face_swap_img_t, face_align_img_t, target_mask_t=face_mask_t, source_mask_t=face_mask_t)
                #print(f"DEBUG: face_swap_img_t shape after RCT: {face_swap_img_t.shape}") # Ожидаем (3, H, W)

            #print("DEBUG: Remapping face_swap")
            expected_swap_output_shape_tuple = face_swap_img_t.shape[:-2] + (frame_height, frame_width)
            #print(f"DEBUG: Calculated expected_swap_output_shape_tuple: {expected_swap_output_shape_tuple}") # Ожидаем (3, frame_H, frame_W)

            frame_face_swap_img_t = lib_cl.remap_np_affine(face_swap_img_t,
                                                           aligned_to_source_uni_mat,
                                                           interpolation=interpolation,
                                                           output_size=(frame_height, frame_width),
                                                           post_op_text='O = clamp(O, 0.0f, 1.0f);',
                                                           expected_o_shape_tuple=expected_swap_output_shape_tuple)

            #print(f"DEBUG: frame_face_swap_img_t shape after remap: {frame_face_swap_img_t.shape}") # Ожидаем (3, frame_H, frame_W)
            #print("DEBUG: Face swap preparation done")
        except Exception as e:
            #print(f"!!!!!!!! ОШИБКА В БЛОКЕ 3 (Подготовка face_swap): {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        try: # --- Блок 4: Финальное слияние ---
            #print("DEBUG: Final merge")
            opacity = np.float32(state.face_opacity)
            frame_face_mask_t_c3 = frame_face_mask_t # Используем результат ремаппинга маски

            #print(f"DEBUG: Current frame shape: {current_frame_image_t.shape}, Mask shape after remap: {frame_face_mask_t.shape}") # Ожидаем маску (1, frame_H, frame_W)

            # --- Проверка и возможное расширение маски ---
            if current_frame_image_t.shape[0] == 3 and frame_face_mask_t.shape[0] == 1:
                #print("DEBUG: Expanding mask to 3 channels using concat")
                # --- ИСПРАВЛЕНИЕ: Используем concat ---
                frame_face_mask_t_c3 = lib_cl.concat([frame_face_mask_t] * 3, axis=0)
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
                #print(f"DEBUG: Mask shape after expand: {frame_face_mask_t_c3.shape}") # Ожидаем (3, frame_H, frame_W)
            elif current_frame_image_t.shape[0] != frame_face_mask_t.shape[0]:
                 #print(f"ERROR: Mismatched channels before merge - Image channels: {current_frame_image_t.shape[0]}, Mask channels: {frame_face_mask_t.shape[0]}")
                 raise ValueError(f"Несовместимые каналы: изображение={current_frame_image_t.shape[0]} против маски={frame_face_mask_t.shape[0]}")
            else:
                 #print(f"DEBUG: Mask channels ({frame_face_mask_t.shape[0]}) match image channels ({current_frame_image_t.shape[0]}). No expansion needed.")
                 frame_face_mask_t_c3 = frame_face_mask_t # Уже правильная форма

            # --- Слияние ---
            if abs(opacity - 1.0) < 1e-6:
                #print("DEBUG: Merging with opacity 1.0")
                merged_frame_t = lib_cl.any_wise('O = I0*(1.0f-I1) + I2*I1', current_frame_image_t, frame_face_mask_t_c3, frame_face_swap_img_t, dtype=np.float32)
            else:
                #print("DEBUG: Merging with opacity < 1.0")
                merged_frame_t = lib_cl.any_wise('O = I0*(1.0f-I1) + I0*I1*(1.0f-I3) + I2*I1*I3', current_frame_image_t, frame_face_mask_t_c3, frame_face_swap_img_t, opacity, dtype=np.float32)
            #print("DEBUG: Merging done")
        except Exception as e:
            #print(f"!!!!!!!! ОШИБКА В БЛОКЕ 4 (Финальное слияние): {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        try: # --- Блок 5: Сжатие цвета ---
            if do_color_compression and state.color_compression > 0:
                #print("DEBUG: Applying color compression")
                color_compression_divisor = max(1.0, (128.0 - state.color_compression))
                compress_op_text = 'float val = ( floor(I0 * I1) / I1 ) + (2.0f / I1); O = clamp(val, 0.0f, 1.0f);'
                merged_frame_t = lib_cl.any_wise(compress_op_text, merged_frame_t, np.float32(color_compression_divisor), dtype=np.float32)
                #print("DEBUG: Color compression done")
            else:
                pass
                 #print("DEBUG: Skipping color compression")
        except Exception as e:
            #print(f"!!!!!!!! ОШИБКА В БЛОКЕ 5 (Сжатие цвета): {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        #print("DEBUG: Exiting _merge_on_gpu successfully")
        return merged_frame_t

    # Основная функция обработки
    def on_tick(self):
        state = self.get_state()

        if self.pending_bcd is None:
            self.start_profile_timing()
            bcd = self.bc_in.read(timeout=0.005)
            if bcd is not None:
                bcd.assign_weak_heap(self.weak_heap)
                frame_image_name = bcd.get_frame_image_name()
                frame_image_bgr_uint8 = bcd.get_image(frame_image_name)
                output_merged_frame = None

                if frame_image_bgr_uint8 is not None:
                    fsi_list = bcd.get_face_swap_info_list()
                    fsi_list_len = len(fsi_list)
                    has_processed_faces = False
                    is_cpu = not isinstance(state.device, lib_cl.DeviceInfo)
                    current_merged_frame = None
                    current_merged_frame_t = None

                    if is_cpu:
                        current_merged_frame = ImageProcessor(frame_image_bgr_uint8).to_ufloat32().get_image('HWC')
                    else:
                        try:
                            if lib_cl.get_default_device() is None:
                                if isinstance(state.device, lib_cl.DeviceInfo):
                                    dev = lib_cl.get_device(state.device)
                                    lib_cl.set_default_device(dev)
                                else:
                                    raise RuntimeError("Выбрано GPU, но устройство по умолчанию не установлено.")
                            current_merged_frame_t = lib_cl.Tensor.from_value(frame_image_bgr_uint8).transpose((2, 0, 1), op_text='O = ((float)I) / 255.0;', dtype=np.float32)
                            #print(f"DEBUG on_tick: Initial frame tensor shape: {current_merged_frame_t.shape}")
                        except Exception as e:
                            #print(f"Ошибка инициализации GPU тензора: {e}. Переход на CPU.")
                            is_cpu = True
                            current_merged_frame = ImageProcessor(frame_image_bgr_uint8).to_ufloat32().get_image('HWC')

                    frame_height, frame_width = frame_image_bgr_uint8.shape[:2]

                    for fsi_id, fsi in enumerate(fsi_list):
                        face_anim_img = bcd.get_image(fsi.face_anim_image_name)
                        if face_anim_img is not None:
                            if is_cpu:
                                current_merged_frame = ImageProcessor(face_anim_img).to_ufloat32().get_image('HWC')
                            else:
                                if current_merged_frame_t is not None:
                                    try:
                                        current_merged_frame_t = lib_cl.Tensor.from_value(face_anim_img, target_tensor=current_merged_frame_t).transpose((2, 0, 1), op_text='O = ((float)I) / 255.0;', dtype=np.float32)
                                        #print(f"DEBUG on_tick: Animation frame tensor shape: {current_merged_frame_t.shape}")
                                    except Exception as e:
                                        #print(f"Ошибка создания анимационного тензора: {e}. Переход на CPU.")
                                        current_merged_frame = ImageProcessor(face_anim_img).to_ufloat32().get_image('HWC')
                                        is_cpu = True
                                else:
                                    current_merged_frame = ImageProcessor(face_anim_img).to_ufloat32().get_image('HWC')
                                    is_cpu = True
                            has_processed_faces = True
                            continue

                        image_to_align_uni_mat = fsi.image_to_align_uni_mat
                        face_resolution = fsi.face_resolution
                        face_align_img = bcd.get_image(fsi.face_align_image_name)
                        face_align_lmrks_mask_img = bcd.get_image(fsi.face_align_lmrks_mask_name)
                        face_align_mask_img = bcd.get_image(fsi.face_align_mask_name)
                        face_swap_img = bcd.get_image(fsi.face_swap_image_name)
                        face_swap_mask_img = bcd.get_image(fsi.face_swap_mask_name)
                        required_data_ok = all_is_not_None(face_resolution, face_swap_img, image_to_align_uni_mat) and \
                                          (not state.face_mask_source or face_align_mask_img is not None) and \
                                          (not state.face_mask_celeb or face_swap_mask_img is not None) and \
                                          (not state.face_mask_lmrks or face_align_lmrks_mask_img is not None) and \
                                          (state.color_transfer != 'rct' or face_align_img is not None)

                        if required_data_ok:
                            has_processed_faces = True
                            aligned_to_source_uni_mat = image_to_align_uni_mat.invert().source_translated(-state.face_x_offset, -state.face_y_offset).source_scaled_around_center(state.face_scale, state.face_scale)
                            aligned_to_source_uni_mat_exact = aligned_to_source_uni_mat.to_exact_mat(face_resolution, face_resolution, frame_width, frame_height)
                            do_color_compression = (fsi_id == fsi_list_len - 1)

                            if is_cpu:
                                current_merged_frame = self._merge_on_cpu(current_merged_frame, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat_exact, frame_width, frame_height, do_color_compression)
                            else:
                                if current_merged_frame_t is not None:
                                    try:
                                        current_merged_frame_t = self._merge_on_gpu(current_merged_frame_t, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat_exact, frame_width, frame_height, do_color_compression)
                                    except Exception as e:
                                        #print(f"Ошибка слияния на GPU: {type(e).__name__}: {e}. Переход на CPU.")
                                        try:
                                            current_merged_frame = current_merged_frame_t.transpose((1, 2, 0)).np()
                                            #print(f"DEBUG on_tick: Converted GPU tensor to CPU numpy shape: {current_merged_frame.shape}")
                                        except Exception as e_np:
                                            #print(f"  Ошибка преобразования тензора в numpy: {e_np}. Использование исходного кадра.")
                                            current_merged_frame = ImageProcessor(frame_image_bgr_uint8).to_ufloat32().get_image('HWC')
                                        is_cpu = True
                                        current_merged_frame = self._merge_on_cpu(current_merged_frame, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat_exact, frame_width, frame_height, do_color_compression)
                                else:
                                    is_cpu = True
                                    current_merged_frame = ImageProcessor(frame_image_bgr_uint8).to_ufloat32().get_image('HWC')
                                    current_merged_frame = self._merge_on_cpu(current_merged_frame, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat_exact, frame_width, frame_height, do_color_compression)

                    if has_processed_faces:
                        if not is_cpu and current_merged_frame_t is not None:
                            try:
                                output_merged_frame = current_merged_frame_t.transpose((1, 2, 0)).np()
                            except Exception as e:
                                #print(f"Ошибка извлечения с GPU: {e}. Использование исходного кадра.")
                                output_merged_frame = ImageProcessor(frame_image_bgr_uint8).to_ufloat32().get_image('HWC')
                        elif is_cpu:
                            output_merged_frame = current_merged_frame
                        else:
                            output_merged_frame = ImageProcessor(frame_image_bgr_uint8).to_ufloat32().get_image('HWC')
                    else:
                         if not is_cpu and current_merged_frame_t is not None:
                              try: output_merged_frame = current_merged_frame_t.transpose((1, 2, 0)).np()
                              except Exception: output_merged_frame = ImageProcessor(frame_image_bgr_uint8).to_ufloat32().get_image('HWC')
                         else:
                              output_merged_frame = ImageProcessor(frame_image_bgr_uint8).to_ufloat32().get_image('HWC')

                    if output_merged_frame is not None:
                        output_merged_frame_uint8 = np.clip(output_merged_frame * 255.0, 0, 255).astype(np.uint8)
                        merged_image_name = f'{frame_image_name}_merged'
                        bcd.set_merged_image_name(merged_image_name)
                        bcd.set_image(merged_image_name, output_merged_frame_uint8)
                    else:
                        bcd.set_merged_image_name(None)

                self.stop_profile_timing()
                self.pending_bcd = bcd

        if self.pending_bcd is not None:
            if self.bc_out.is_full_read(1):
                self.bc_out.write(self.pending_bcd)
                self.pending_bcd = None
            else:
                time.sleep(0.001)


class Sheet:
    class Host(lib_csw.Sheet.Host):
        def __init__(self):
            super().__init__()
            self.device = lib_csw.DynamicSingleSwitch.Client()
            self.face_x_offset = lib_csw.Number.Client()
            self.face_y_offset = lib_csw.Number.Client()
            self.face_scale = lib_csw.Number.Client()
            self.face_mask_source = lib_csw.Flag.Client()
            self.face_mask_celeb = lib_csw.Flag.Client()
            self.face_mask_lmrks = lib_csw.Flag.Client()
            self.face_mask_x_offset = lib_csw.Number.Client()
            self.face_mask_y_offset = lib_csw.Number.Client()
            self.face_mask_scale = lib_csw.Number.Client()
            self.face_mask_erode = lib_csw.Number.Client()
            self.face_mask_blur = lib_csw.Number.Client()
            self.color_transfer = lib_csw.DynamicSingleSwitch.Client()
            self.interpolation = lib_csw.DynamicSingleSwitch.Client()
            self.color_compression = lib_csw.Number.Client()
            self.face_opacity = lib_csw.Number.Client()

    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.device = lib_csw.DynamicSingleSwitch.Host()
            self.face_x_offset = lib_csw.Number.Host()
            self.face_y_offset = lib_csw.Number.Host()
            self.face_scale = lib_csw.Number.Host()
            self.face_mask_source = lib_csw.Flag.Host()
            self.face_mask_celeb = lib_csw.Flag.Host()
            self.face_mask_lmrks = lib_csw.Flag.Host()
            self.face_mask_x_offset = lib_csw.Number.Host()
            self.face_mask_y_offset = lib_csw.Number.Host()
            self.face_mask_scale = lib_csw.Number.Host()
            self.face_mask_erode = lib_csw.Number.Host()
            self.face_mask_blur = lib_csw.Number.Host()
            self.color_transfer = lib_csw.DynamicSingleSwitch.Host()
            self.interpolation = lib_csw.DynamicSingleSwitch.Host()
            self.color_compression = lib_csw.Number.Host()
            self.face_opacity = lib_csw.Number.Host()


class WorkerState(BackendWorkerState):
    device = None  # 'CPU' или lib_cl.DeviceInfo
    face_x_offset: float = 0.0
    face_y_offset: float = 0.0
    face_scale: float = 1.0
    face_mask_source: bool = True
    face_mask_celeb: bool = True
    face_mask_lmrks: bool = False
    face_mask_x_offset: float = 0.0
    face_mask_y_offset: float = 0.0
    face_mask_scale: float = 1.0
    face_mask_erode: int = 5
    face_mask_blur: int = 25
    color_transfer: str = 'rct'
    interpolation: str = 'bilinear'
    color_compression: float = 1.0
    face_opacity: float = 1.0