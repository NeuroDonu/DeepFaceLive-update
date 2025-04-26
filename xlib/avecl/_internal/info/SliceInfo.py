# C:\Users\neurodonu\Downloads\DeepFaceLive\DeepFaceLive\xlib\avecl\_internal\info\SliceInfo.py:
import math
import numpy as np
from ..AShape import AShape

class SliceInfo:
    __slots__ = ['o_shape', 'o_shape_kd', 'just_reshaped','axes_bes','axes_abs_bes']

    def __init__(self, shape : AShape, slices_in): # Переименовал входной параметр для ясности
        """
        Slice info.
        can raise ValueError,TypeError during the construction
        """
        # --- Начало предварительной обработки срезов (как было) ---
        processed_slices = []
        before_ellipsis = None

        # Убедимся, что slices_in - это кортеж или список
        if not isinstance(slices_in, (tuple, list)):
             _slices_in_iterable = (slices_in,)
        else:
             _slices_in_iterable = slices_in

        for s in _slices_in_iterable:
            if s is Ellipsis:
                before_ellipsis = processed_slices
                processed_slices = []
                continue
            # Проверяем типы, которые МЫ ожидаем после hashable_slices ИЛИ стандартные типы
            elif s is not None and not isinstance(s, (int, tuple, type(Ellipsis), slice)): # Добавили slice и Ellipsis на всякий случай
                 # Кортежи проверяем позже на длину
                 if isinstance(s, tuple):
                     pass # Разрешаем кортежи, проверим позже
                 else:
                      raise ValueError(f'unknown slice argument {s} of type {s.__class__}')
            processed_slices.append(s)

        if before_ellipsis is not None:
            # Обработка Ellipsis
            new_slices_n_axes = sum([1 for x in processed_slices if x is not None])
            before_ellipsis_n_axes = sum([1 for x in before_ellipsis if x is not None])
            # Заполняем (None, None, None) - это нормально, будет обработано как 3-элементный кортеж
            processed_slices = before_ellipsis + \
                               [(None,None,None)]*max(0, shape.ndim-before_ellipsis_n_axes-new_slices_n_axes) + \
                               processed_slices

        new_slices_n_axes = sum([ 1 for x in processed_slices if x is not None])
        if new_slices_n_axes > shape.ndim:
            raise ValueError('slices arguments more than shape axes')
        elif new_slices_n_axes < shape.ndim:
            # Дополняем до нужной размерности (None, None, None)
            processed_slices += [(None,None,None)]*( shape.ndim - new_slices_n_axes )

        slices = tuple(processed_slices) # Это финальный кортеж срезов для обработки
        # --- Конец предварительной обработки срезов ---


        # --- Начало основного цикла обработки (с исправлениями) ---
        output_is_reshaped = True
        o_shape = []
        o_shape_kd = []
        axes_bes = []
        axes_abs_bes = []
        i_axis = 0

        for i_v, v in enumerate(slices): # Идем по обработанным срезам
            if v is None:
                o_shape.append(1)
                continue # None создает новую ось размерности 1

            i_axis_size = shape[i_axis]
            current_axis_idx = i_axis # Сохраним для сообщений об ошибках
            i_axis += 1

            _b, _e, _s = None, None, None # Временные переменные
            is_int_slice = False

            if isinstance(v, int):
                # Целочисленный индекс
                _b, _e, _s = v, v, 0 # Шаг 0 означает выбор одного элемента
                is_int_slice = True
            elif isinstance(v, tuple):
                # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
                if len(v) == 4 and v[0] is slice:
                    # Это кортеж (slice, start, stop, step) от hashable_slices
                    _, _b, _e, _s = v # Распаковываем 4, игнорируем первый
                elif len(v) == 3:
                    # Это кортеж (b, e, s), например, от Ellipsis -> (None, None, None)
                    _b, _e, _s = v
                else:
                    # Неожиданная длина кортежа
                    raise ValueError(f'Slice tuple for axis {current_axis_idx} has unexpected length {len(v)}: {repr(v)}')
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
            # --- Можно добавить обработку 'slice' объекта, если hashable_slices вдруг его пропустит ---
            elif isinstance(v, slice):
                 #print(f"ПРЕДУПРЕЖДЕНИЕ: SliceInfo получил необработанный объект slice: {v}. Обработка как (_b=v.start, _e=v.stop, _s=v.step)")
                 _b, _e, _s = v.start, v.stop, v.step
            else:
                # Неожиданный тип после всех проверок
                raise TypeError(f"Unexpected type in final processed slices for axis {current_axis_idx}: {type(v)}")

            # Теперь нормализуем _b, _e, _s в b, e, s
            b, e, s = _b, _e, _s

            # Проверка шага (только если это не был int индекс)
            if not is_int_slice and s == 0:
                 # Шаг 0 от int индекса - это нормально, означает выбор одного элемента
                 # Но если это пришло из slice или кортежа, то это ошибка
                 if not (_b == _e and _s == 0): # Дополнительная проверка, что это не случай b=e, s=0
                     raise ValueError(f'Slice step cannot be zero for axis {current_axis_idx}')

            # ---- Начало нормализации b, e, s (как было) ----
            if s is None: s = 1
            if b is None: b = 0 if s >= 0 else i_axis_size-1
            if e is None:
                e = i_axis_size if s >= 0 else -1 # -1 для отрицательного шага означает "до самого начала"
            # Коррекция отрицательных индексов (кроме e = -1 при s < 0)
            if isinstance(e, int) and e < 0 and not (s < 0 and e == -1): e += i_axis_size
            if isinstance(b, int) and b < 0: b += i_axis_size

            # Приведение к границам оси
            if s >= 0:
                b = np.clip(b, 0, i_axis_size)
                e = np.clip(e, 0, i_axis_size)
                # Убрали проверку b > e, так как clip может сделать b == e, что даст пустой срез - это нормально
            else: # s < 0
                b = np.clip(b, -1, i_axis_size-1) # Позволяем b быть -1 (для совместимости?)
                e = np.clip(e, -1, i_axis_size)   # Позволяем e быть -1
                # Убрали проверку b <= e, так как clip может сделать b == e, что даст пустой срез

            # Вычисление абсолютных значений для использования в ядре (как было)
            abs_s = abs(s if s != 0 else 1)
            if s >= 0:
                abs_b, abs_e = b, e
            else: # s < 0
                 # Вычисляем реальные границы для отрицательного шага
                 if b < e: # Если после клиппинга начало стало меньше конца (пустой срез)
                     count = 0
                 else:
                     count = math.ceil( (b - e) / abs_s )

                 if count <= 0:
                     abs_b, abs_e = b + 1, b + 1 # Пустой диапазон
                 else:
                     abs_b = b - (count - 1) * abs_s
                     abs_e = b + 1
            # ---- Конец нормализации b, e, s ----


            axes_bes.append((b,e,s))
            axes_abs_bes.append((abs_b, abs_e, abs_s))

            # Проверка, изменился ли срез (как было)
            if output_is_reshaped and i_axis_size != 1 and not (b == 0 and e == i_axis_size and s == 1):
                output_is_reshaped = False

            # Вычисление размера выходной оси (как было)
            if s == 0: # Индекс
                o_axis_size = 1 if (isinstance(b, int) and 0 <= b < i_axis_size) else 0
            elif s > 0:
                 o_axis_size = math.ceil( max(0, e - b) / s )
            else: # s < 0
                 o_axis_size = math.ceil( max(0, b - e) / abs_s )

            # Добавление размера оси в выходную форму (если ось не была удалена индексом)
            if not is_int_slice:
                 if o_axis_size >= 1:
                    o_shape.append(o_axis_size)
            # Добавление в форму с сохранением размерности
            o_shape_kd.append(max(1, o_axis_size)) # Всегда добавляем, но размер 1, если ось сжалась/была индексом

        self.just_reshaped = output_is_reshaped and tuple(shape.shape) == tuple(o_shape_kd) # Уточнил условие
        self.o_shape = AShape(o_shape)
        self.o_shape_kd = AShape(o_shape_kd)
        self.axes_bes = tuple(axes_bes) # Преобразуем в кортежи для неизменяемости
        self.axes_abs_bes = tuple(axes_abs_bes)