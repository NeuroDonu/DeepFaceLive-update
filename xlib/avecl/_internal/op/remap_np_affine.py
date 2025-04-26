# C:\Users\neurodonu\Downloads\DeepFaceLive\DeepFaceLive\xlib\avecl\_internal\op\remap_np_affine.py
import numpy as np

from ..AShape import AShape
from ..backend import Kernel
from ..EInterpolation import EInterpolation
from ..HKernel import HKernel
from ..SCacheton import SCacheton
from ..Tensor import Tensor

# Добавляем expected_o_shape_tuple в сигнатуру
def remap_np_affine (input_t : Tensor, affine_n : np.ndarray, interpolation : EInterpolation = None, inverse=False, output_size=None, post_op_text=None, dtype=None, expected_o_shape_tuple : tuple = None) -> Tensor:
    """
    remap affine operator for all channels using single numpy affine mat

    arguments

        input_t     Tensor (...,H,W) - Ожидается формат с каналами в начале (C,H,W) или (N,C,H,W) для GPU операций

        affine_n    np.array (2,3)

        interpolation    EInterpolation

        post_op_text    cl kernel
                        post operation with output float value named 'O'
                        example 'O = 2*O;'

        output_size     (height, width) - ВАЖНО: порядок (H, W)

        dtype           Выходной тип данных

        expected_o_shape_tuple(None) Кортеж ожидаемой выходной формы, например (C, OH, OW) или (N, C, OH, OW).
                                     Используется для обхода проблем с кэшированием _RemapAffineOp.
    """
    if affine_n.shape != (2,3):
        raise ValueError('affine_n.shape must be (2,3)')

    # Передаем expected_o_shape_tuple в ключ SCacheton
    op = SCacheton.get(_RemapAffineOp, input_t.shape, input_t.dtype, interpolation, output_size, post_op_text, dtype, expected_o_shape_tuple)

    # Проверка формы, которую теперь хранит op
    try:
        calculated_shape_tuple = op.o_shape.shape
        #print(f"DEBUG remap_np_affine: Op has op.o_shape: {calculated_shape_tuple}")
        if expected_o_shape_tuple and calculated_shape_tuple != expected_o_shape_tuple:
             #print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             #print(f"WARNING remap_np_affine: op.o_shape {calculated_shape_tuple} != expected_o_shape_tuple {expected_o_shape_tuple}")
             #print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             # Можно принудительно использовать ожидаемую форму, если мы ей доверяем больше
             calculated_shape_tuple = expected_o_shape_tuple
    except Exception as e:
        #print(f"DEBUG remap_np_affine: Error accessing op.o_shape.shape: {e}. op.o_shape: {op.o_shape}")
        # Если не можем прочитать форму из op, но ожидаемая есть, используем ее
        calculated_shape_tuple = expected_o_shape_tuple

    # Создаем выходной тензор, используя финальную (возможно, скорректированную) форму
    final_shape_tuple = calculated_shape_tuple
    if final_shape_tuple is None:
        # Если форма так и не определена, это ошибка
        raise ValueError(f"Cannot determine output shape for remap_np_affine. Input shape: {input_t.shape}, output_size: {output_size}, expected_o_shape: {expected_o_shape_tuple}")

    output_t = Tensor( final_shape_tuple, op.o_dtype, device=input_t.get_device() )

    #print(f"DEBUG remap_np_affine: Created output_t with actual shape: {output_t.shape}") # Лог реальной формы

    # Инвертируем матрицу, если нужно (поведение по умолчанию как в cv2.warpAffine)
    ((a, b, c),
     (d, e, f)) = affine_n
    if not inverse:
        D = a*e - b*d
        D = 1.0 / D if D != 0.0 else 0.0
        a, b, c, d, e, f = (  e*D, -b*D, (b*f-e*c)*D ,
                             -d*D,  a*D, (d*c-a*f)*D )

    # Запускаем ядро OpenCL
    input_t.get_device().run_kernel(op.forward_krn, output_t.get_buffer(), input_t.get_buffer(),
                                    np.float32(a), np.float32(b), np.float32(c), np.float32(d), np.float32(e), np.float32(f) )

    return output_t


# Конструктор теперь тоже принимает expected_o_shape_tuple
class _RemapAffineOp():
    def __init__(self, i_shape : AShape, i_dtype, interpolation, o_size, post_op_text, o_dtype, expected_o_shape_tuple : tuple = None):
        # Основная логика конструктора
        if np.dtype(i_dtype).type == np.bool_:
            raise ValueError('np.bool_ dtype of i_dtype is not supported.')
        # !!! ВАЖНО: avecl обычно работает с форматом (..., H, W), где ... - это КАНАЛЫ или батчи+каналы
        # Поэтому ndim должен быть >= 2
        if i_shape.ndim < 2:
            raise ValueError(f'i_shape.ndim must be >= 2 (expected format like [N,C,]H,W), got {i_shape.ndim} for shape {i_shape.shape}')
        if interpolation is None:
            interpolation = EInterpolation.LINEAR

        #print(f"DEBUG _RemapAffineOp.__init__: Received i_shape: {i_shape.shape}, ndim: {i_shape.ndim}, expected_o_shape_tuple: {expected_o_shape_tuple}")

        # Используем ПЕРЕДАННУЮ форму или вычисляем, если ее нет
        if expected_o_shape_tuple is not None:
            # Проверяем, что ожидаемая форма имеет хотя бы 2 измерения (H, W)
            if len(expected_o_shape_tuple) < 2:
                 raise ValueError(f"expected_o_shape_tuple {expected_o_shape_tuple} must have at least 2 dimensions (H, W)")
            #print(f"DEBUG _RemapAffineOp: Using provided expected_o_shape_tuple: {expected_o_shape_tuple}")
            final_o_shape_tuple = expected_o_shape_tuple
            # Извлекаем OH, OW из ожидаемой формы для использования в ядре (если o_size не задан)
            OH, OW = final_o_shape_tuple[-2:]
        else:
            # Вычисляем форму, если она не была передана
            #print(f"DEBUG _RemapAffineOp: expected_o_shape_tuple is None, calculating shape manually...")
            IH,IW = i_shape.shape[-2:]
            if o_size is not None:
                 if len(o_size) == 2:
                      OH, OW = o_size[0], o_size[1] # Порядок (H, W)
                      #print(f"DEBUG _RemapAffineOp: Using output_size (H, W): ({OH}, {OW})")
                 else:
                      raise ValueError(f"output_size должен быть кортежем из 2 элементов (height, width), получено: {o_size}")
            else:
                OH,OW = IH,IW
                #print(f"DEBUG _RemapAffineOp: Using input_size (H, W) as output_size: ({OH}, {OW})")

            # --- Логика ручного вычисления формы ---
            o_shape_list = []
            if i_shape.ndim > 2:
                leading_dims_tuple = i_shape.shape[:-2]
                #print(f"DEBUG _RemapAffineOp (Manual): Adding leading dims: {leading_dims_tuple}")
                o_shape_list.extend(list(leading_dims_tuple))
            #print(f"DEBUG _RemapAffineOp (Manual): Adding target H, W: ({OH}, {OW})")
            o_shape_list.extend([OH, OW])
            final_o_shape_tuple = tuple(o_shape_list)
            #print(f"DEBUG _RemapAffineOp: Manually calculated shape: {final_o_shape_tuple}")
        # --- Конец вычисления/использования формы ---

        self.o_shape = AShape(final_o_shape_tuple) # Сохраняем финальную форму
        #print(f"DEBUG _RemapAffineOp: Set self.o_shape to: {self.o_shape.shape}")

        self.o_dtype = o_dtype = o_dtype if o_dtype is not None else i_dtype
        if post_op_text is None: post_op_text = ''

        # --- Генерация текста ядра OpenCL ---
        # Используем финальную self.o_shape для определения размерности и индексов
        # Важно: HKernel.define_tensor и HKernel.decompose_idx_to_axes_idxs используют self.o_shape
        # Ядро ожидает, что последние два измерения self.o_shape - это H и W
        # Ядро также неявно использует размерности входного тензора i_shape (Im1 = W, Im2 = H)
        kernel_ndim = self.o_shape.ndim
        if kernel_ndim < 2:
             raise ValueError(f"Cannot generate kernel: Output shape {self.o_shape.shape} has less than 2 dimensions.")

        common_defines = f"""
{HKernel.define_tensor('O', self.o_shape, o_dtype)}
{HKernel.define_tensor('I', i_shape, i_dtype)}
"""
        kernel_args = f"""
__kernel void impl(__global O_PTR_TYPE* O_PTR_NAME, __global const I_PTR_TYPE* I_PTR_NAME,
                   float a, float b, float c,
                   float d, float e, float f)
{{
    size_t gid = get_global_id(0);
    {HKernel.decompose_idx_to_axes_idxs('gid', 'O', kernel_ndim)}
    // om{kernel_ndim-1} = Индекс по последнему измерению (W)
    // om{kernel_ndim-2} = Индекс по предпоследнему измерению (H)
    // Ведущие измерения (om0, om1, ...) используются в HKernel.axes_seq_enum
"""

        if interpolation == EInterpolation.LINEAR:
            kernel_text = common_defines + kernel_args + f"""

    float cx01 = om{kernel_ndim-1}*a + om{kernel_ndim-2}*b + c; // Координата X во входном изображении
    float cy01 = om{kernel_ndim-1}*d + om{kernel_ndim-2}*e + f; // Координата Y во входном изображении

    float cx0f = floor(cx01);   int cx0 = (int)cx0f;
    float cy0f = floor(cy01);   int cy0 = (int)cy0f;
    float cx1f = cx0f+1;        int cx1 = (int)cx1f;
    float cy1f = cy0f+1;        int cy1 = (int)cy1f;

    // Читаем 4 пикселя из входного изображения I, используя вычисленные координаты (cx0, cy0) и т.д.
    // и сохраняя ведущие измерения (om0, om1, ...), если они есть.
    float p00 = I_GLOBAL_LOAD(I_IDX_MOD({HKernel.axes_seq_enum('O', kernel_ndim-2, suffix='cy0,cx0')}));
    float p01 = I_GLOBAL_LOAD(I_IDX_MOD({HKernel.axes_seq_enum('O', kernel_ndim-2, suffix='cy0,cx1')}));
    float p10 = I_GLOBAL_LOAD(I_IDX_MOD({HKernel.axes_seq_enum('O', kernel_ndim-2, suffix='cy1,cx0')}));
    float p11 = I_GLOBAL_LOAD(I_IDX_MOD({HKernel.axes_seq_enum('O', kernel_ndim-2, suffix='cy1,cx1')}));

    // Проверка выхода за границы входного изображения (Im1=W, Im2=H)
    p00 *= (cx1f - cx01)*(cy1f - cy01)*(cy0 >= 0 & cy0 < Im2 & cx0 >= 0 & cx0 < Im1);
    p01 *= (cx01 - cx0f)*(cy1f - cy01)*(cy0 >= 0 & cy0 < Im2 & cx1 >= 0 & cx1 < Im1);
    p10 *= (cx1f - cx01)*(cy01 - cy0f)*(cy1 >= 0 & cy1 < Im2 & cx0 >= 0 & cx0 < Im1);
    p11 *= (cx01 - cx0f)*(cy01 - cy0f)*(cy1 >= 0 & cy1 < Im2 & cx1 >= 0 & cx1 < Im1);

    // Билинейная интерполяция
    float O = p00 + p01 + p10 + p11;
    {post_op_text} // Применяем пост-обработку, если есть
    O_GLOBAL_STORE(gid, O); // Записываем результат в выходной тензор O
}}
"""
        elif interpolation == EInterpolation.CUBIC:
            kernel_text = common_defines + """
float cubic(float p0, float p1, float p2, float p3, float x) {
    float a0 = p1;
    float a1 = p2 - p0;
    float a2 = 2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3;
    float a3 = 3.0f * (p1 - p2) + p3 - p0;
    return a0 + 0.5f * x * (a1 + x * (a2 + x * a3));
}
""" + kernel_args + f"""
    float cx01f = om{kernel_ndim-1}*a + om{kernel_ndim-2}*b + c;
    float cy01f = om{kernel_ndim-1}*d + om{kernel_ndim-2}*e + f;

    float cxf = floor(cx01f);   int cx = (int)cxf;
    float cyf = floor(cy01f);   int cy = (int)cyf;

    float dx = cx01f-cxf;
    float dy = cy01f-cyf;

    float row[4];

    #pragma unroll
    for (int y=cy-1, j=0; y<=cy+2; y++, j++) {{
        float col[4];
        #pragma unroll
        for (int x=cx-1, i=0; x<=cx+2; x++, i++) {{
            float sxy = I_GLOBAL_LOAD(I_IDX_MOD({HKernel.axes_seq_enum('O', kernel_ndim-2, suffix='y,x')}));
            col[i] = sxy*(y >= 0 & y < Im2 & x >= 0 & x < Im1);
        }}
        row[j] = cubic(col[0], col[1], col[2], col[3], dx);
    }}

    float O = cubic(row[0], row[1], row[2], row[3], dy);
    {post_op_text}
    O_GLOBAL_STORE(gid, O);
}}
"""
        elif interpolation in [EInterpolation.LANCZOS3, EInterpolation.LANCZOS4]:
            RAD = 3 if interpolation == EInterpolation.LANCZOS3 else 4
            kernel_text = common_defines + kernel_args + f"""
    float cx01f = om{kernel_ndim-1}*a + om{kernel_ndim-2}*b + c;
    float cy01f = om{kernel_ndim-1}*d + om{kernel_ndim-2}*e + f;

    float cxf = floor(cx01f);   int cx = (int)cxf;
    float cyf = floor(cy01f);   int cy = (int)cyf;

    #define RAD {RAD}
    #define M_PI_F 3.14159265358979323846f // Определим константу PI

    float Fy[2 * RAD];
    float Fx[2 * RAD];

    #pragma unroll
    for (int y=cy-RAD+1, j=0; y<=cy+RAD; y++, j++) {{
        float dy = fabs(cy01f - (float)y); // Явное приведение y к float
        if (dy < 1e-4f) Fy[j] = 1.0f;
        else if (dy >= (float)RAD) Fy[j] = 0.0f; // >= чтобы избежать деления на 0 в знаменателе
        else Fy[j] = ( (float)RAD * sin(M_PI_F * dy) * sin(M_PI_F * dy / (float)RAD) ) / ( (M_PI_F*M_PI_F)*dy*dy );
    }}

    #pragma unroll
    for (int x=cx-RAD+1, i=0; x<=cx+RAD; x++, i++) {{
        float dx = fabs(cx01f - (float)x); // Явное приведение x к float
        if (dx < 1e-4f) Fx[i] = 1.0f;
        else if (dx >= (float)RAD) Fx[i] = 0.0f;
        else Fx[i] = ( (float)RAD * sin(M_PI_F * dx) * sin(M_PI_F * dx / (float)RAD) ) / ( (M_PI_F*M_PI_F)*dx*dx );
    }}

    float FxFysum = 0.0f;
    float O = 0.0f;

    #pragma unroll
    for (int y=cy-RAD+1, j=0; y<=cy+RAD; y++, j++) {{
        #pragma unroll
        for (int x=cx-RAD+1, i=0; x<=cx+RAD; x++, i++) {{
            float sxy = I_GLOBAL_LOAD(I_IDX_MOD({HKernel.axes_seq_enum('O', kernel_ndim-2, suffix='y,x')}));
            float Fxyv = Fx[i]*Fy[j];
            FxFysum += Fxyv;
            // Умножаем на значение пикселя только если он в пределах границ
            O += sxy*Fxyv*(y >= 0 & y < Im2 & x >= 0 & x < Im1);
        }}
    }}

    if (fabs(FxFysum) > 1e-6f) O = O / FxFysum; else O = 0.0f; // Избегаем деления на ноль
    {post_op_text}
    O_GLOBAL_STORE(gid, O);
}}
"""
        else:
            raise ValueError(f'Unsupported interpolation type {interpolation}')

        # Компилируем ядро
        self.forward_krn = Kernel(global_shape=(self.o_shape.size,), kernel_text=kernel_text)