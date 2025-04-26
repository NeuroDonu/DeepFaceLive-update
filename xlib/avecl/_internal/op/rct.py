# C:\Users\neurodonu\Downloads\DeepFaceLive\DeepFaceLive\xlib\avecl\_internal\op\rct.py
import numpy as np

from ..Tensor import Tensor
from .any_wise import any_wise, sqrt
from .concat import concat
from .cvt_color import cvt_color
from .slice_ import split
from .reduce import moments
from .reshape import reshape # Импортируем функцию reshape

def rct(target_t : Tensor, source_t : Tensor, target_mask_t : Tensor = None, source_mask_t : Tensor = None, mask_cutoff = 0.5) -> Tensor:
    """
    Transfer color using rct method.
    # ... (docstring как был) ...
    """
    if target_t.shape != source_t.shape:
        raise ValueError(f"rct: target_t shape {target_t.shape} != source_t shape {source_t.shape}")

    is_batched = target_t.ndim == 4
    if is_batched:
        ch_axis = 1
        spatial_axes = (2, 3)
        mask_transpose_axes = (0, 3, 1, 2) # N, H, W, C=1 -> N, C=1, H, W
        reshape_target_stat_shape = (target_t.shape[0], target_t.shape[1], 1, 1) # (N, C, 1, 1)
    else: # ndim == 3
        ch_axis = 0
        spatial_axes = (1, 2)
        mask_transpose_axes = (2, 0, 1) # H, W, C=1 -> C=1, H, W
        reshape_target_stat_shape = (target_t.shape[0], 1, 1) # (C, 1, 1)

    if target_t.shape[ch_axis] != 3:
        raise ValueError(f"rct: Input tensors must have 3 channels (BGR), but got {target_t.shape[ch_axis]} for target_t")

    # --- Конвертация в LAB ---
    target_t = cvt_color(target_t, 'BGR', 'LAB', ch_axis=ch_axis)
    source_t = cvt_color(source_t, 'BGR', 'LAB', ch_axis=ch_axis)

    # --- Подготовка тензоров для статистики (применение маски) ---
    target_stat_t = target_t
    if target_mask_t is not None:
        expected_mask_shape_suffix = target_t.shape[spatial_axes[0]:] + (1,)
        if is_batched: expected_mask_shape = (target_t.shape[0],) + expected_mask_shape_suffix
        else: expected_mask_shape = expected_mask_shape_suffix
        if target_mask_t.shape != expected_mask_shape:
             raise ValueError(f"rct: target_mask_t shape {target_mask_t.shape} is not compatible with target_t shape {target_t.shape}. Expected mask shape {expected_mask_shape}")
        target_mask_t_bc = target_mask_t.transpose(mask_transpose_axes)
        target_stat_t = any_wise('O = I0*(I1 >= I2)', target_t, target_mask_t_bc, np.float32(mask_cutoff) )

    source_stat_t = source_t
    if source_mask_t is not None:
        expected_mask_shape_suffix = source_t.shape[spatial_axes[0]:] + (1,)
        if is_batched: expected_mask_shape = (source_t.shape[0],) + expected_mask_shape_suffix
        else: expected_mask_shape = expected_mask_shape_suffix
        if source_mask_t.shape != expected_mask_shape:
             raise ValueError(f"rct: source_mask_t shape {source_mask_t.shape} is not compatible with source_t shape {source_t.shape}. Expected mask shape {expected_mask_shape}")
        source_mask_t_bc = source_mask_t.transpose(mask_transpose_axes)
        source_stat_t = any_wise('O = I0*(I1 >= I2)', source_t, source_mask_t_bc, np.float32(mask_cutoff) )

    # --- Вычисление моментов (БЕЗ keepdims) ---
    # Результат будет формы (N, C) или (C,)
    target_stat_mean_t, target_stat_var_t = moments(target_stat_t, axes=spatial_axes)
    source_stat_mean_t, source_stat_var_t = moments(source_stat_t, axes=spatial_axes)

    # --- ИЗМЕНЕНИЕ: Reshape статистики для broadcast ---
    # Преобразуем (N, C) -> (N, C, 1, 1) или (C,) -> (C, 1, 1)
    # print(f"DEBUG rct: Reshaping stats to target shape: {reshape_target_stat_shape}") # Можно раскомментировать
    target_stat_mean_t = reshape(target_stat_mean_t, reshape_target_stat_shape)
    target_stat_var_t  = reshape(target_stat_var_t,  reshape_target_stat_shape)
    source_stat_mean_t = reshape(source_stat_mean_t, reshape_target_stat_shape)
    source_stat_var_t  = reshape(source_stat_var_t,  reshape_target_stat_shape)
    # print(f"DEBUG rct: Shape after reshape - target_mean: {target_stat_mean_t.shape}") # Можно раскомментировать

    # --- Предотвращение деления на ноль или слишком маленькое значение дисперсии ---
    min_var = np.float32(1.0 / (255.0 * 255.0))
    target_stat_var_t = any_wise('O = fmax(I0, I1)', target_stat_var_t, min_var)
    source_stat_var_t = any_wise('O = fmax(I0, I1)', source_stat_var_t, min_var)

    # --- Применение формулы RCT ---
    target_t = any_wise(f"""
O_0 = clamp( (I0_0 - I1_0) * sqrt(I2_0) / sqrt(I3_0) + I4_0, 0.0, 100.0); /* L channel */
O_1 = clamp( (I0_1 - I1_1) * sqrt(I2_1) / sqrt(I3_1) + I4_1, -127.0, 127.0); /* A channel */
O_2 = clamp( (I0_2 - I1_2) * sqrt(I2_2) / sqrt(I3_2) + I4_2, -127.0, 127.0); /* B channel */
""", target_t,           # I0 - target LAB image
     target_stat_mean_t, # I1 - target mean
     source_stat_var_t,  # I2 - source variance
     target_stat_var_t,  # I3 - target variance
     source_stat_mean_t, # I4 - source mean
     dim_wise_axis=ch_axis) # Применяем по каналам

    # --- Конвертация обратно в BGR ---
    return cvt_color(target_t, 'LAB', 'BGR', ch_axis=ch_axis)