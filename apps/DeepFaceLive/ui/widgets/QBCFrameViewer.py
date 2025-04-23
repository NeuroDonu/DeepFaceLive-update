from localization import L
from resources.fonts import QXFontDB
from xlib import qt as qtx

from ... import backend


class QBCFrameViewer(qtx.QXCollapsibleSection):
    def __init__(self,  backed_weak_heap : backend.BackendWeakHeap,
                        bc : backend.BackendConnection,
                        preview_width=256):
        self._timer = qtx.QXTimer(interval=16, timeout=self._on_timer_16ms, start=True)

        self._backed_weak_heap = backed_weak_heap
        self._bc = bc
        self._bcd_id = None

        layered_images = self._layered_images = qtx.QXFixedLayeredImages(preview_width, preview_width)
        info_label = self._info_label = qtx.QXLabel( font=QXFontDB.get_fixedwidth_font(size=7))

        main_l = qtx.QXVBoxLayout([ (layered_images, qtx.AlignCenter),
                                    (info_label, qtx.AlignCenter), ])
        super().__init__(title=L('@QBCFrameViewer.title'), content_layout=main_l)

    def _on_timer_16ms(self):
        top_qx = self.get_top_QXWindow()
        if not self.is_opened() or (top_qx is not None and top_qx.is_minimized()):
            # Если секция закрыта или окно свернуто, просто прочитаем все,
            # чтобы освободить буфер, но ничего не будем отображать.
            while self._bc.read(timeout=0) is not None:
                 pass # Прочитать и выбросить, чтобы освободить место
            self._bcd_id = self._bc.get_write_id() # Обновить ID, чтобы не читать снова сразу после открытия
            return

        last_bcd = None
        while True:
            # Читаем следующий доступный кадр из соединения (неблокирующий вызов)
            # read() должен удалять прочитанный элемент из буфера bc
            bcd = self._bc.read(timeout=0)

            if bcd is None:
                # Нет больше новых данных в буфере на данный момент
                break
            else:
                # Есть новые данные, запомним их для отображения
                # Убедимся, что он связан с кучей для управления памятью
                bcd.assign_weak_heap(self._backed_weak_heap)
                last_bcd = bcd # Запоминаем последний успешный кадр
                # self._bcd_id = bcd.get_uid() # Обновляем ID последнего обработанного (если get_uid есть)
        
        if last_bcd is not None:
            self._layered_images.clear_images()
            frame_image_name = last_bcd.get_frame_image_name()
            frame_image = last_bcd.get_image(frame_image_name)

            if frame_image is not None:
                self._layered_images.add_image(frame_image)
                h, w = frame_image.shape[:2]
                self._info_label.setText(f'{frame_image_name} {w}x{h}')
        
    def clear(self):
        self._layered_images.clear_images()
