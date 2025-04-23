import numpy as np
from localization import L
from resources.fonts import QXFontDB
from xlib import qt as qtx

from ... import backend


class QBCFaceAlignViewer(qtx.QXCollapsibleSection):
    def __init__(self,  backed_weak_heap : backend.BackendWeakHeap,
                        bc : backend.BackendConnection,
                        preview_width=256,):

        self._preview_width = preview_width
        self._timer = qtx.QXTimer(interval=16, timeout=self._on_timer_16ms, start=True)
        self._backed_weak_heap = backed_weak_heap
        self._bc = bc
        self._bcd_id = None # Можно будет использовать для сравнения UID, если get_uid будет у BCD

        layered_images = self._layered_images = qtx.QXFixedLayeredImages(preview_width, preview_width)
        info_label = self._info_label = qtx.QXLabel( font=QXFontDB.get_fixedwidth_font(size=7))

        super().__init__(title=L('@QBCFaceAlignViewer.title'),
                         content_layout=qtx.QXVBoxLayout([(layered_images, qtx.AlignCenter),
                                                          (info_label, qtx.AlignCenter)])  )

    def _on_timer_16ms(self):
        top_qx = self.get_top_QXWindow()
        if not self.is_opened() or (top_qx is not None and top_qx.is_minimized()):
            # Если секция закрыта или окно свернуто, читаем и выбрасываем
            while self._bc.read(timeout=0) is not None:
                 pass
            # self._bcd_id = self._bc.get_write_id() # Можно обновить ID
            return

        # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
        last_bcd = None # Храним последний прочитанный BCD
        while True:
            # Читаем следующий доступный кадр из соединения
            bcd = self._bc.read(timeout=0) # Используем правильный read()

            if bcd is None:
                # Нет больше новых данных в буфере
                break
            else:
                # Есть новые данные, запомним их
                bcd.assign_weak_heap(self._backed_weak_heap)
                last_bcd = bcd # Запоминаем

        # Отображаем данные ТОЛЬКО из самого последнего кадра (last_bcd)
        if last_bcd is not None:
            # Опционально: проверка ID, если необходимо
            # current_bcd_id = last_bcd.get_uid()
            # if self._bcd_id != current_bcd_id:
            #    self._bcd_id = current_bcd_id
            #    # Код обновления ниже выполнять только если ID изменился

            self._layered_images.clear_images() # Очищаем предыдущие слои
            found_data = False
            for fsi in last_bcd.get_face_swap_info_list():
                # Ищем выровненное лицо в последнем bcd
                face_image = last_bcd.get_image (fsi.face_align_image_name)
                if face_image is not None:
                    h,w = face_image.shape[:2]
                    self._layered_images.add_image(face_image)

                    # Рисуем лендмарки и прямоугольник, если они есть
                    if fsi.face_align_ulmrks is not None:
                        lmrks_layer = np.zeros( (self._preview_width, self._preview_width, 4), dtype=np.uint8)
                        fsi.face_align_ulmrks.draw(lmrks_layer, (0,255,0,255)) # Зеленые точки

                        if fsi.face_urect is not None and fsi.image_to_align_uni_mat is not None:
                            aligned_uni_rect = fsi.face_urect.transform(fsi.image_to_align_uni_mat)
                            aligned_uni_rect.draw(lmrks_layer, (0,0,255,255) ) # Красный прямоугольник?

                        self._layered_images.add_image(lmrks_layer) # Добавляем слой с лендмарками

                    self._info_label.setText(f'{w}x{h}')
                    found_data = True
                    break # Отображаем только первое найденное выровненное лицо

            if not found_data:
                # Если в последнем BCD не нашлось выровненного лица, очищаем
                self._layered_images.clear_images()
                self._info_label.setText("")
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    def clear(self):
        self._layered_images.clear_images()