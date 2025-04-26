from typing import List

import numpy as np

from ..AShape import AShape
from ..AAxes import AAxes

class BroadcastInfo:
    """
    Broadcast info for multi shapes.
    # ... (остальная часть docstring) ...
    """

    __slots__ = ['o_shape', 'br_shapes', 'shapes_tiles', 'shapes_reduction_axes']

    def __init__(self, shapes : List[AShape]):
        # --- ДОБАВЛЕНО ДЛЯ ОТЛАДКИ ---
        # Печатаем список форм, полученных конструктором
        #print(f"DEBUG BroadcastInfo.__init__: Received shapes: {[s.shape for s in shapes]}")
        # --- КОНЕЦ ОТЛАДКИ ---

        if not shapes: # Проверка на пустой список
             raise ValueError('shape_list is empty')

        highest_rank = sorted([shape.ndim for shape in shapes])[-1]

        # Broadcasted all shapes to highest ndim with (1,)-s in from left side
        # AShape handles adding dimensions internally if needed? Let's adjust.
        br_shapes_tuples = []
        for shape in shapes:
             # Дополняем единицами слева до highest_rank
             padding = (1,) * (highest_rank - shape.ndim)
             br_shapes_tuples.append( padding + shape.shape ) # Используем shape.shape кортеж
        br_shapes = [AShape(s) for s in br_shapes_tuples] # Создаем AShape объекты

        # Determine o_shape from all shapes
        # Используем NumPy для broadcast_shapes для надежности
        try:
            o_shape_np = np.broadcast_shapes(*(s.shape for s in br_shapes))
            o_shape = AShape(o_shape_np)
        except ValueError as e:
             # Если NumPy не может совместить, то и мы не сможем
             shapes_str = ', '.join(str(s.shape) for s in shapes)
             br_shapes_str = ', '.join(str(s.shape) for s in br_shapes)
             print(f"ERROR BroadcastInfo: NumPy broadcast failed for initial shapes: {shapes_str} -> broadcasted shapes: {br_shapes_str}")
             raise ValueError(f'Operands could not be broadcast together. Initial shapes: {shapes_str} -> Broadcasted shapes: {br_shapes_str}. NumPy error: {e}')
        shapes_tiles = []
        shapes_reduction_axes = []
        for br_shape in br_shapes:
            # Сравнение br_shape и o_shape должно работать, т.к. они теперь одной размерности
            if br_shape.ndim != o_shape.ndim:
                 # Этого не должно происходить после выравнивания размерности
                 raise RuntimeError(f"Internal BroadcastInfo Error: Mismatched ranks after alignment ({br_shape.ndim} vs {o_shape.ndim})")

            shape_tile = []
            shape_reduction_axes = []
            for axis, (br_dim, o_dim) in enumerate(zip(br_shape.shape, o_shape.shape)):
                if br_dim == o_dim:
                    shape_tile.append(1)
                elif br_dim == 1 and o_dim != 1:
                    shape_tile.append(o_dim)
                    shape_reduction_axes.append(axis)
                # elif br_dim != 1 and o_dim == 1: # Этот случай невозможен при правильном broadcast
                #     shape_tile.append(1)
                else:
                    # Этот случай тоже не должен возникать, если o_shape рассчитан верно
                    print(f"ERROR BroadcastInfo: Mismatch detected during tile calculation: br_shape={br_shape.shape}, o_shape={o_shape.shape}, axis={axis}, br_dim={br_dim}, o_dim={o_dim}")
                    raise ValueError(f'operands could not be broadcast together (mismatch at axis {axis}): {br_shape.shape} vs {o_shape.shape}')

            shapes_tiles.append(tuple(shape_tile)) # Используем кортежи
            shapes_reduction_axes.append( AAxes(shape_reduction_axes) )


        self.o_shape : AShape = o_shape # Уже AShape
        self.br_shapes : List[AShape] = br_shapes
        self.shapes_tiles : List[tuple] = shapes_tiles # Список кортежей
        self.shapes_reduction_axes : List [AAxes] = shapes_reduction_axes