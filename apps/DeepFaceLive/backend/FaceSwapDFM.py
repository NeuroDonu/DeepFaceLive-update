import time
from pathlib import Path
from typing import Dict

import numpy as np
from modelhub import DFLive
from xlib import os as lib_os
from xlib.image.ImageProcessor import ImageProcessor
from xlib.mp import csw as lib_csw
from xlib.python import all_is_not_None

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendSignal, BackendWeakHeap, BackendWorker,
                          BackendWorkerState)


class FaceSwapDFM(BackendHost):
    def __init__(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, dfm_models_path : Path, backend_db : BackendDB = None,
                  id : int = 0):
        self._id = id
        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=FaceSwapDFMWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out, dfm_models_path])

    def get_control_sheet(self) -> 'Sheet.Host': return super().get_control_sheet()

    def _get_name(self):
        return super()._get_name()# + f'{self._id}'

class FaceSwapDFMWorker(BackendWorker):
    def get_state(self) -> 'WorkerState': return super().get_state()
    def get_control_sheet(self) -> 'Sheet.Worker': return super().get_control_sheet()

    def on_start(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, dfm_models_path : Path):
        self.weak_heap = weak_heap
        self.reemit_frame_signal = reemit_frame_signal
        self.bc_in = bc_in
        self.bc_out = bc_out
        self.dfm_models_path = dfm_models_path

        self.pending_bcd = None

        self.dfm_model_initializer = None
        self.dfm_model = None

        lib_os.set_timer_resolution(1)

        state, cs = self.get_state(), self.get_control_sheet()

        cs.model.call_on_selected(self.on_cs_model)
        cs.device.call_on_selected(self.on_cs_device)
        cs.swap_all_faces.call_on_flag(self.on_cs_swap_all_faces)
        cs.face_id.call_on_number(self.on_cs_face_id)
        cs.morph_factor.call_on_number(self.on_cs_morph_factor)
        cs.presharpen_amount.call_on_number(self.on_cs_presharpen_amount)
        cs.pre_gamma_red.call_on_number(self.on_cs_pre_gamma_red)
        cs.pre_gamma_green.call_on_number(self.on_cs_pre_gamma_green)
        cs.pre_gamma_blue.call_on_number(self.on_cs_pre_gamma_blue)
        cs.post_gamma_red.call_on_number(self.on_cs_post_gamma_red)
        cs.post_gamma_blue.call_on_number(self.on_cs_post_gamma_blue)
        cs.post_gamma_green.call_on_number(self.on_cs_post_gamma_green)
        cs.two_pass.call_on_flag(self.on_cs_two_pass)

        cs.device.enable()
        cs.device.set_choices( DFLive.get_available_devices(), none_choice_name='@misc.menu_select')
        cs.device.select(state.device)

    def on_cs_device(self, idx, device):
        state, cs = self.get_state(), self.get_control_sheet()
        if device is not None and state.device == device:
            cs.model.enable()
            cs.model.set_choices( DFLive.get_available_models_info(self.dfm_models_path), none_choice_name='@misc.menu_select')
            cs.model.select(state.model)
        else:
            state.device = device
            self.save_state()
            self.restart()

    def on_cs_model(self, idx, selected_model_info : DFLive.DFMModelInfo):
        state, cs = self.get_state(), self.get_control_sheet()

        if selected_model_info is not None:
            # Определяем, НУЖНО ли запускать/перезапускать инициализацию
            needs_init = False
            if state.model != selected_model_info:
                # Модель изменилась - точно нужна инициализация
                ##print(f"DFM: Model changed to '{selected_model_info.get_name()}'. Needs init.")
                needs_init = True
            elif self.dfm_model is None and self.dfm_model_initializer is None:
                # Выбрана та же модель, но она не загружена и не инициализируется
                # (вероятно, после перезапуска или сбоя предыдущей попытки)
                ##print(f"DFM: Re-attempting initialization for '{selected_model_info.get_name()}'. Needs init.")
                needs_init = True
            # Дополнительно: Можно добавить кнопку "Force Re-Init" и проверять ее флаг здесь,
            # если понадобится принудительный перезапуск без смены модели.

            if needs_init:
                # --- Код инициализации ---
                ##print(f"DFM: Starting initialization process for '{selected_model_info.get_name()}'...")

                # 1. Обновляем состояние
                state.model = selected_model_info
                state.model_state = state.models_state.get(selected_model_info.get_name(), ModelState())
                state.models_state[selected_model_info.get_name()] = state.model_state

                # 2. Очищаем предыдущие ресурсы (ВАЖНО!)
                if self.dfm_model is not None:
                    ##print("DFM: Unloading previous model.")
                    # !!! Если у модели есть метод выгрузки, вызвать его ЗДЕСЬ !!!
                    # например: self.dfm_model.unload() или .release() или .dispose()
                    # Если такого метода нет, просто обнуляем ссылку
                    self.dfm_model = None
                if self.dfm_model_initializer is not None:
                    ##print("DFM: Cancelling previous initializer.")
                    # !!! Если у инициализатора есть метод отмены, вызвать его ЗДЕСЬ !!!
                    # например: self.dfm_model_initializer.cancel()
                    # Если нет, просто обнуляем
                    self.dfm_model_initializer = None

                # 3. Отключаем контролы на время инициализации
                cs.model_info_label.disable()
                cs.swap_all_faces.disable()
                cs.face_id.disable()
                cs.morph_factor.disable()
                cs.presharpen_amount.disable()
                cs.pre_gamma_red.disable()
                cs.pre_gamma_green.disable()
                cs.pre_gamma_blue.disable()
                cs.post_gamma_red.disable()
                cs.post_gamma_blue.disable()
                cs.post_gamma_green.disable()
                cs.two_pass.disable()
                cs.model_dl_error.disable()
                cs.model_dl_error.set_error(None)
                cs.model_dl_progress.disable() # Отключим и прогресс на всякий случай

                # 4. Создаем НОВЫЙ инициализатор
                #print(f"DFM: Creating initializer on device '{state.device}'...")
                try:
                    # Используем state.model, который мы только что обновили
                    self.dfm_model_initializer = DFLive.DFMModel_from_info(state.model, state.device)
                except Exception as e:
                    #print(f"DFM: ***** EXCEPTION during DFLive.DFMModel_from_info creation *****")
                    import traceback
                    traceback.print_exc()
                    self.dfm_model_initializer = None

                # 5. Устанавливаем статус busy
                if self.dfm_model_initializer is not None:
                    #print("DFM: Initializer created. Setting busy.")
                    self.set_busy(True)
                else:
                    #print("DFM: ERROR - Failed to create model initializer!")
                    self.set_busy(False)
                    cs.model_dl_error.enable()
                    cs.model_dl_error.set_error("Failed to create model initializer.")

                self.save_state() # Сохраняем выбор модели
            else:
                # Модель та же, и она либо загружена, либо уже инициализируется
                #print(f"DFM: Model '{selected_model_info.get_name()}' is already loaded or initializing. No action needed.")
                # Просто убедимся, что статус busy правильный
                self.set_busy(self.dfm_model is None and self.dfm_model_initializer is not None)

    def on_cs_swap_all_faces(self, swap_all_faces):
        state, cs = self.get_state(), self.get_control_sheet()
        model_state = state.model_state
        if model_state is not None:
            model_state.swap_all_faces = swap_all_faces

            if not swap_all_faces:
                cs.face_id.enable()
                cs.face_id.set_config(lib_csw.Number.Config(min=0, max=999, step=1, decimals=0, allow_instant_update=True))
                cs.face_id.set_number(state.model_state.face_id if state.model_state.face_id is not None else 0)
            else:
                cs.face_id.disable()

            self.save_state()
            self.reemit_frame_signal.send()


    def on_cs_face_id(self, face_id):
        state, cs = self.get_state(), self.get_control_sheet()
        model_state = state.model_state
        if model_state is not None:
            cfg = cs.face_id.get_config()
            face_id = model_state.face_id = int(np.clip(face_id, cfg.min, cfg.max))
            cs.face_id.set_number(face_id)
            self.save_state()
            self.reemit_frame_signal.send()

    def on_cs_presharpen_amount(self, presharpen_amount):
        state, cs = self.get_state(), self.get_control_sheet()
        model_state = state.model_state
        if model_state is not None:
            cfg = cs.presharpen_amount.get_config()
            presharpen_amount = model_state.presharpen_amount = float(np.clip(presharpen_amount, cfg.min, cfg.max))
            cs.presharpen_amount.set_number(presharpen_amount)
            self.save_state()
            self.reemit_frame_signal.send()

    def on_cs_morph_factor(self, morph_factor):
        state, cs = self.get_state(), self.get_control_sheet()
        model_state = state.model_state
        if model_state is not None:
            cfg = cs.morph_factor.get_config()
            morph_factor = model_state.morph_factor = float(np.clip(morph_factor, cfg.min, cfg.max))
            cs.morph_factor.set_number(morph_factor)
            self.save_state()
            self.reemit_frame_signal.send()

    def on_cs_pre_gamma_red(self, pre_gamma_red):
        state, cs = self.get_state(), self.get_control_sheet()
        model_state = state.model_state
        if model_state is not None:
            cfg = cs.pre_gamma_red.get_config()
            pre_gamma_red = model_state.pre_gamma_red = float(np.clip(pre_gamma_red, cfg.min, cfg.max))
            cs.pre_gamma_red.set_number(pre_gamma_red)
            self.save_state()
            self.reemit_frame_signal.send()

    def on_cs_pre_gamma_green(self, pre_gamma_green):
        state, cs = self.get_state(), self.get_control_sheet()
        model_state = state.model_state
        if model_state is not None:
            cfg = cs.pre_gamma_green.get_config()
            pre_gamma_green = model_state.pre_gamma_green = float(np.clip(pre_gamma_green, cfg.min, cfg.max))
            cs.pre_gamma_green.set_number(pre_gamma_green)
            self.save_state()
            self.reemit_frame_signal.send()

    def on_cs_pre_gamma_blue(self, pre_gamma_blue):
        state, cs = self.get_state(), self.get_control_sheet()
        model_state = state.model_state
        if model_state is not None:
            cfg = cs.pre_gamma_blue.get_config()
            pre_gamma_blue = model_state.pre_gamma_blue = float(np.clip(pre_gamma_blue, cfg.min, cfg.max))
            cs.pre_gamma_blue.set_number(pre_gamma_blue)
            self.save_state()
            self.reemit_frame_signal.send()

    def on_cs_post_gamma_red(self, post_gamma_red):
        state, cs = self.get_state(), self.get_control_sheet()
        model_state = state.model_state
        if model_state is not None:
            cfg = cs.post_gamma_red.get_config()
            post_gamma_red = model_state.post_gamma_red = float(np.clip(post_gamma_red, cfg.min, cfg.max))
            cs.post_gamma_red.set_number(post_gamma_red)
            self.save_state()
            self.reemit_frame_signal.send()

    def on_cs_post_gamma_blue(self, post_gamma_blue):
        state, cs = self.get_state(), self.get_control_sheet()
        model_state = state.model_state
        if model_state is not None:
            cfg = cs.post_gamma_blue.get_config()
            post_gamma_blue = model_state.post_gamma_blue = float(np.clip(post_gamma_blue, cfg.min, cfg.max))
            cs.post_gamma_blue.set_number(post_gamma_blue)
            self.save_state()
            self.reemit_frame_signal.send()

    def on_cs_post_gamma_green(self, post_gamma_green):
        state, cs = self.get_state(), self.get_control_sheet()
        model_state = state.model_state
        if model_state is not None:
            cfg = cs.post_gamma_green.get_config()
            post_gamma_green = model_state.post_gamma_green = float(np.clip(post_gamma_green, cfg.min, cfg.max))
            cs.post_gamma_green.set_number(post_gamma_green)
            self.save_state()
            self.reemit_frame_signal.send()

    def on_cs_two_pass(self, two_pass):
        state, cs = self.get_state(), self.get_control_sheet()
        model_state = state.model_state
        if model_state is not None:
            model_state.two_pass = two_pass
            self.save_state()
            self.reemit_frame_signal.send()

    def on_tick(self):
        state, cs = self.get_state(), self.get_control_sheet()

        if self.dfm_model_initializer is not None:
            try:
                events = self.dfm_model_initializer.process_events()
                # Печатаем ВСЕ атрибуты объекта events, чтобы увидеть, что он возвращает
                #print(f"DFM Init Tick: Events = {vars(events)}")
            except Exception as e:
                # Ловим возможные ошибки при вызове process_events
                #print(f"DFM Init Tick: ***** EXCEPTION during process_events() *****")
                import traceback
                traceback.print_exc()
                # Возможно, стоит остановить инициализацию при ошибке здесь
                self.dfm_model_initializer = None
                self.set_busy(False)
                cs.model_dl_error.enable()
                cs.model_dl_error.set_error(f"Error in process_events: {e}")
                return # Выходим из on_tick, так как инициализатор сломался

            if events.prev_status_downloading:
                self.set_busy(True)
                cs.model_dl_progress.disable()

            if events.new_status_downloading:
                self.set_busy(False)
                cs.model_dl_progress.enable()
                cs.model_dl_progress.set_config( lib_csw.Progress.Config(title='@FaceSwapDFM.downloading_model') )
                cs.model_dl_progress.set_progress(0)

            elif events.new_status_initialized:
                ##print("DFM Init Tick: Model initialized.")
                self.dfm_model = events.dfm_model
                self.dfm_model_initializer = None

                model_width, model_height = self.dfm_model.get_input_res()

                cs.model_info_label.enable()
                cs.model_info_label.set_config( lib_csw.InfoLabel.Config(info_icon=True,
                                                    info_lines=[f'@FaceSwapDFM.model_information',
                                                                '',
                                                                f'@FaceSwapDFM.filename',
                                                                f'{self.dfm_model.get_model_path().name}',
                                                                '',
                                                                f'@FaceSwapDFM.resolution',
                                                                f'{model_width}x{model_height}']) )

                cs.swap_all_faces.enable()
                cs.swap_all_faces.set_flag( state.model_state.swap_all_faces if state.model_state.swap_all_faces is not None else False)

                if self.dfm_model.has_morph_value():
                    cs.morph_factor.enable()
                    cs.morph_factor.set_config(lib_csw.Number.Config(min=0, max=1, step=0.01, decimals=2, allow_instant_update=True))
                    cs.morph_factor.set_number(state.model_state.morph_factor if state.model_state.morph_factor is not None else 0.75)

                cs.presharpen_amount.enable()
                cs.presharpen_amount.set_config(lib_csw.Number.Config(min=0, max=10, step=0.1, decimals=1, allow_instant_update=True))
                cs.presharpen_amount.set_number(state.model_state.presharpen_amount if state.model_state.presharpen_amount is not None else 0)

                cs.pre_gamma_red.enable()
                cs.pre_gamma_red.set_config(lib_csw.Number.Config(min=0.01, max=4, step=0.01, decimals=2, allow_instant_update=True))
                cs.pre_gamma_red.set_number(state.model_state.pre_gamma_red if state.model_state.pre_gamma_red is not None else 1)

                cs.pre_gamma_green.enable()
                cs.pre_gamma_green.set_config(lib_csw.Number.Config(min=0.01, max=4, step=0.01, decimals=2, allow_instant_update=True))
                cs.pre_gamma_green.set_number(state.model_state.pre_gamma_green if state.model_state.pre_gamma_green is not None else 1)

                cs.pre_gamma_blue.enable()
                cs.pre_gamma_blue.set_config(lib_csw.Number.Config(min=0.010, max=4, step=0.01, decimals=2, allow_instant_update=True))
                cs.pre_gamma_blue.set_number(state.model_state.pre_gamma_blue if state.model_state.pre_gamma_blue is not None else 1)

                cs.post_gamma_red.enable()
                cs.post_gamma_red.set_config(lib_csw.Number.Config(min=0.010, max=4, step=0.01, decimals=2, allow_instant_update=True))
                cs.post_gamma_red.set_number(state.model_state.post_gamma_red if state.model_state.post_gamma_red is not None else 1)

                cs.post_gamma_blue.enable()
                cs.post_gamma_blue.set_config(lib_csw.Number.Config(min=0.010, max=4, step=0.01, decimals=2, allow_instant_update=True))
                cs.post_gamma_blue.set_number(state.model_state.post_gamma_blue if state.model_state.post_gamma_blue is not None else 1)

                cs.post_gamma_green.enable()
                cs.post_gamma_green.set_config(lib_csw.Number.Config(min=0.010, max=4, step=0.01, decimals=2, allow_instant_update=True))
                cs.post_gamma_green.set_number(state.model_state.post_gamma_green if state.model_state.post_gamma_green is not None else 1)

                cs.two_pass.enable()
                cs.two_pass.set_flag(state.model_state.two_pass if state.model_state.two_pass is not None else False)

                self.set_busy(False)
                self.reemit_frame_signal.send()

            elif events.new_status_error:
                self.set_busy(False)
                cs.model_dl_error.enable()
                cs.model_dl_error.set_error(events.error)

            if events.download_progress is not None:
                cs.model_dl_progress.set_progress(events.download_progress)

        if self.pending_bcd is None:
            self.start_profile_timing()

            bcd = self.bc_in.read(timeout=0.005)
            if bcd is not None:
                bcd.assign_weak_heap(self.weak_heap)

                model_state = state.model_state
                dfm_model = self.dfm_model
                #print(f"DFM Process Tick: Checking model. dfm_model is None? {dfm_model is None}. model_state is None? {model_state is None}")
                if all_is_not_None(dfm_model, model_state):
                    #print("  DFM Process Tick: Model and state OK. Entering face loop.")
                    try:
                        for i, fsi in enumerate(bcd.get_face_swap_info_list()):
                            #print(f"    DFM: Checking face idx {i}. Swap all? {model_state.swap_all_faces}. Target ID? {model_state.face_id}")
                            if not model_state.swap_all_faces and model_state.face_id != i:
                                #print(f"      DFM: Skipping face idx {i}.")
                                continue

                            # --- Шаг 1: Получение изображения ---
                            #print(f"      DFM: [Step 1] Attempting get_image('{fsi.face_align_image_name}')...")
                            face_align_image = None
                            try:
                                face_align_image = bcd.get_image(fsi.face_align_image_name)
                            except Exception as e_getimg:
                                #print(f"      DFM: [Step 1] ***** EXCEPTION during bcd.get_image() *****")
                                traceback.print_exc(); continue
                            if face_align_image is None:
                                #print(f"      DFM: [Step 1] ERROR! face_align_image is None!")
                                continue
                            #print(f"      DFM: [Step 1] OK. Shape: {face_align_image.shape}")

                            # --- Шаг 2: Препроцессинг ---
                            #print(f"      DFM: [Step 2] Preprocessing...")
                            try:
                                pre_gamma_red = model_state.pre_gamma_red; pre_gamma_green = model_state.pre_gamma_green; pre_gamma_blue = model_state.pre_gamma_blue
                                fai_ip = ImageProcessor(face_align_image)
                                if model_state.presharpen_amount != 0: fai_ip.gaussian_sharpen(sigma=1.0, power=model_state.presharpen_amount)
                                if pre_gamma_red != 1.0 or pre_gamma_green != 1.0 or pre_gamma_blue != 1.0: fai_ip.gamma(pre_gamma_red, pre_gamma_green, pre_gamma_blue)
                                face_align_image = fai_ip.get_image('HWC') # Получаем результат препроцессинга
                                if face_align_image is None: raise ValueError("Preprocessing resulted in None image")
                            except Exception as e_preproc:
                                #print(f"      DFM: [Step 2] ***** EXCEPTION during preprocessing *****")
                                traceback.print_exc(); continue
                            #print(f"      DFM: [Step 2] OK. Shape after: {face_align_image.shape}")

                            # --- Шаг 3: Вызов модели ---
                            #print(f"      DFM: [Step 3] Calling dfm_model.convert...")
                            convert_result = None
                            try:
                                convert_result = dfm_model.convert(face_align_image, morph_factor=model_state.morph_factor)
                            except Exception as e_convert:
                                #print(f"      DFM: [Step 3] ***** EXCEPTION during dfm_model.convert() *****")
                                traceback.print_exc(); continue
                            #print(f"      DFM: [Step 3] OK. Result type: {type(convert_result)}")

                            # --- Шаг 4: Обработка результата модели ---
                            #print(f"      DFM: [Step 4] Processing convert result...")
                            try:
                                if isinstance(convert_result, tuple) and len(convert_result) == 3:
                                    celeb_face, celeb_face_mask_img, face_align_mask_img = convert_result
                                    #print(f"        DFM: Result shapes: face={celeb_face.shape if celeb_face is not None else None}, mask={celeb_face_mask_img.shape if celeb_face_mask_img is not None else None}")
                                    if celeb_face is not None and celeb_face_mask_img is not None:
                                        celeb_face, celeb_face_mask_img, face_align_mask_img = celeb_face[0], celeb_face_mask_img[0], face_align_mask_img[0]
                                        #print(f"        DFM: Unpacked shapes: face={celeb_face.shape}, mask={celeb_face_mask_img.shape}")
                                    else:
                                        #print("        DFM: ERROR - celeb_face or mask is None in result tuple!")
                                        continue # Пропускаем, если результат None
                                else:
                                    #print(f"        DFM: ERROR - Unexpected convert result format!")
                                    continue # Пропускаем, если формат не тот
                            except Exception as e_procres:
                                #print(f"      DFM: [Step 4] ***** EXCEPTION during result processing *****")
                                traceback.print_exc(); continue
                            #print(f"      DFM: [Step 4] OK.")

                            # --- Шаг 5: Постпроцессинг ---
                            #print(f"      DFM: [Step 5] Postprocessing...")
                            try:
                                if model_state.two_pass:
                                    #print("        DFM: Applying two_pass...")
                                    celeb_face2, celeb_face_mask_img2, _ = dfm_model.convert(celeb_face, morph_factor=model_state.morph_factor)
                                    celeb_face, celeb_face_mask_img = celeb_face2[0], celeb_face_mask_img2[0]
                                post_gamma_red = model_state.post_gamma_red; post_gamma_blue = model_state.post_gamma_blue; post_gamma_green = model_state.post_gamma_green
                                if post_gamma_red != 1.0 or post_gamma_blue != 1.0 or post_gamma_green != 1.0:
                                    #print("        DFM: Applying post_gamma...")
                                    celeb_face = ImageProcessor(celeb_face).gamma(post_gamma_red, post_gamma_blue, post_gamma_green).get_image('HWC')
                                if celeb_face is None: raise ValueError("Postprocessing resulted in None image")
                            except Exception as e_postproc:
                                #print(f"      DFM: [Step 5] ***** EXCEPTION during postprocessing *****")
                                traceback.print_exc(); continue
                            #print(f"      DFM: [Step 5] OK. Final shape: {celeb_face.shape}")

                            # --- Шаг 6: Сохранение в BCD ---
                            #print(f"      DFM: [Step 6] Setting images in BCD...")
                            try:
                                fsi.face_align_mask_name = f'{fsi.face_align_image_name}_mask'
                                fsi.face_swap_image_name = f'{fsi.face_align_image_name}_swapped'
                                fsi.face_swap_mask_name  = f'{fsi.face_swap_image_name}_mask'
                                bcd.set_image(fsi.face_align_mask_name, face_align_mask_img)
                                bcd.set_image(fsi.face_swap_image_name, celeb_face)
                                bcd.set_image(fsi.face_swap_mask_name, celeb_face_mask_img)
                            except Exception as e_setimg:
                                #print(f"      DFM: [Step 6] ***** EXCEPTION during bcd.set_image() *****")
                                traceback.print_exc(); continue # Пропускаем, если сохранить не удалось
                            #print(f"      DFM: [Step 6] OK. Images set for face idx {i}.")

                        # Конец блока if face_align_image is not None:
                    # --- Конец цикла for ---
                        #print("  DFM Process Tick: Finished face loop.")
                    except Exception as e_loop:
                        #print(f"    DFM: ***** UNCAUGHT EXCEPTION during face processing loop (outer try) *****")
                        traceback.print_exc()

                else:
                     if self.dfm_model_initializer is None: # Печатаем причину пропуска, только если не инициализируется
                         #print(f"  DFM Process Tick: Skipping swap. Reason: dfm_model is None? {dfm_model is None}, model_state is None? {model_state is None}")
                         pass
                # --- Конец if all_is_not_None ---

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
            self.model = lib_csw.DynamicSingleSwitch.Client()
            self.model_info_label = lib_csw.InfoLabel.Client()
            self.model_dl_progress = lib_csw.Progress.Client()
            self.model_dl_error = lib_csw.Error.Client()
            self.device = lib_csw.DynamicSingleSwitch.Client()
            self.swap_all_faces = lib_csw.Flag.Client()
            self.face_id = lib_csw.Number.Client()
            self.morph_factor = lib_csw.Number.Client()
            self.presharpen_amount = lib_csw.Number.Client()
            self.pre_gamma_red = lib_csw.Number.Client()
            self.pre_gamma_green = lib_csw.Number.Client()
            self.pre_gamma_blue = lib_csw.Number.Client()
            self.post_gamma_red = lib_csw.Number.Client()
            self.post_gamma_blue = lib_csw.Number.Client()
            self.post_gamma_green = lib_csw.Number.Client()
            self.two_pass = lib_csw.Flag.Client()

    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.model = lib_csw.DynamicSingleSwitch.Host()
            self.model_info_label = lib_csw.InfoLabel.Host()
            self.model_dl_progress = lib_csw.Progress.Host()
            self.model_dl_error = lib_csw.Error.Host()
            self.device = lib_csw.DynamicSingleSwitch.Host()
            self.swap_all_faces = lib_csw.Flag.Host()
            self.face_id = lib_csw.Number.Host()
            self.morph_factor = lib_csw.Number.Host()
            self.presharpen_amount = lib_csw.Number.Host()
            self.pre_gamma_red = lib_csw.Number.Host()
            self.pre_gamma_green = lib_csw.Number.Host()
            self.pre_gamma_blue = lib_csw.Number.Host()
            self.post_gamma_red = lib_csw.Number.Host()
            self.post_gamma_blue = lib_csw.Number.Host()
            self.post_gamma_green = lib_csw.Number.Host()
            self.two_pass = lib_csw.Flag.Host()

class ModelState(BackendWorkerState):
    swap_all_faces : bool = None
    face_id : int = None
    morph_factor : float = None
    presharpen_amount : float = None
    pre_gamma_red : float = None
    pre_gamma_blue : float = None
    pre_gamma_green: float = None
    post_gamma_red : float = None
    post_gamma_blue : float = None
    post_gamma_green : float = None
    two_pass : bool = None

class WorkerState(BackendWorkerState):
    def __init__(self):
        super().__init__()
        self.device = None
        self.model : DFLive.DFMModelInfo = None
        self.models_state : Dict[str, ModelState] = {}
        self.model_state : ModelState = None
