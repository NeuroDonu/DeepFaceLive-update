from localization import L
from resources.fonts import QXFontDB
from xlib import qt as qtx

from ... import backend


class QBCFaceSwapViewer(qtx.QXCollapsibleSection):
    """
    """
    def __init__(self,  backed_weak_heap : backend.BackendWeakHeap,
                        bc : backend.BackendConnection,
                        preview_width=256,):
        self._preview_width = preview_width
        self._timer = qtx.QXTimer(interval=16, timeout=self._on_timer_16ms, start=True)

        self._backed_weak_heap = backed_weak_heap
        self._bc = bc
        self._bcd_id = None

        layered_images = self._layered_images = qtx.QXFixedLayeredImages(preview_width, preview_width)
        info_label = self._info_label = qtx.QXLabel( font=QXFontDB.get_fixedwidth_font(size=7))

        main_l = qtx.QXVBoxLayout([ (layered_images, qtx.AlignCenter),
                                    (info_label, qtx.AlignCenter) ])

        super().__init__(title=L('@QBCFaceSwapViewer.title'), content_layout=main_l)


    def _on_timer_16ms(self):
        top_qx = self.get_top_QXWindow()
        if not self.is_opened() or (top_qx is not None and top_qx.is_minimized()):
            # Если секция закрыта или окно свернуто, читаем и выбрасываем, чтобы освободить буфер
            while self._bc.read(timeout=0) is not None:
                 pass
            # self._bcd_id = self._bc.get_write_id() # Можно обновить ID, если нужно будет сравнивать
            return

        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        last_bcd = None # Храним последний прочитанный BCD
        while True:
            # Читаем следующий доступный кадр из соединения (неблокирующий вызов)
            bcd = self._bc.read(timeout=0) # Используем правильный метод read()

            if bcd is None:
                # Нет больше новых данных в буфере
                break
            else:
                # Есть новые данные, запомним их
                bcd.assign_weak_heap(self._backed_weak_heap)
                last_bcd = bcd # Запоминаем

        # Отображаем данные ТОЛЬКО из самого последнего кадра (last_bcd)
        if last_bcd is not None:
            # Здесь можно добавить проверку ID, если необходимо (last_bcd.get_uid() и self._bcd_id)
            # чтобы обновлять только если ID изменился

            self._layered_images.clear_images() # Очищаем предыдущие слои
            found_image = False
            for fsi in last_bcd.get_face_swap_info_list():
                # Ищем замененное лицо в последнем bcd
                face_swap_image = last_bcd.get_image(fsi.face_swap_image_name)
                if face_swap_image is not None:
                    self._layered_images.add_image(face_swap_image)
                    h, w = face_swap_image.shape[0:2]
                    self._info_label.setText(f'{w}x{h}')
                    found_image = True
                    break # Отображаем только первое найденное замененное лицо из списка fsi

            if not found_image:
                # Если в последнем BCD не нашлось изображения лица, очищаем
                self._layered_images.clear_images()
                self._info_label.setText("")
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    def clear(self):
        self._layered_images.clear_images()
