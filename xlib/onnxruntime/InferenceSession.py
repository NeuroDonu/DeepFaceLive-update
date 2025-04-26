import onnx
import onnxruntime as rt
from io import BytesIO
from .device import ORTDeviceInfo


def InferenceSession_with_device(onnx_model_or_path, device_info : ORTDeviceInfo):
    """
    Construct onnxruntime.InferenceSession with this Device.

     device_info     ORTDeviceInfo

    can raise Exception
    """

    if isinstance(onnx_model_or_path, onnx.ModelProto):
        b = BytesIO()
        onnx.save(onnx_model_or_path, b)
        onnx_model_or_path = b.getvalue()

    device_ep = device_info.get_execution_provider()
    if device_ep not in rt.get_available_providers():
        raise Exception(f'{device_ep} is not avaiable in onnxruntime')

    # --- НАСТРОЙКА ОПЦИЙ СЕССИИ ---
    sess_options = rt.SessionOptions()

    # Отключаем pattern для DML (оставляем как было)
    if device_ep == 'DmlExecutionProvider':
        sess_options.enable_mem_pattern = False

    # --- НАСТРОЙКА ОПЦИЙ ПРОВАЙДЕРА (EP) ---
    providers_list = []
    provider_options_list = []

    if device_ep == 'TensorrtExecutionProvider':
        #print("Configuring TensorRT Execution Provider...") # Отладочное сообщение
        trt_options = {
            'device_id': device_info.get_index(),
            'trt_engine_cache_enable': False,  # <--- ОТКЛЮЧАЕМ КЭШ ДВИЖКА TRT
            'trt_fp16_enable': False,      # <--- ОТКЛЮЧАЕМ FP16 (используем FP32)
            # 'trt_int8_enable': False,    # Можно добавить, если вдруг используется INT8
            # 'trt_max_workspace_size': 2*1024*1024*1024, # Можно задать лимит памяти (например, 2GB)
            # 'trt_engine_cache_path': '/path/to/cache', # Убедитесь, что не используется, если cache_enable=False
        }
        providers_list.append(device_ep)
        provider_options_list.append(trt_options)
        # Рекомендуется добавлять CPU EP как запасной вариант
        if 'CPUExecutionProvider' in rt.get_available_providers():
            providers_list.append('CPUExecutionProvider')
            provider_options_list.append({}) # Пустые опции для CPU

    elif device_ep == 'CUDAExecutionProvider':
        #print("Configuring CUDA Execution Provider...") # Отладочное сообщение
        cuda_options = {
            'device_id': device_info.get_index(),
            # 'arena_extend_strategy': 'kSameAsRequested', # Пример других опций CUDA
        }
        providers_list.append(device_ep)
        provider_options_list.append(cuda_options)
        if 'CPUExecutionProvider' in rt.get_available_providers():
            providers_list.append('CPUExecutionProvider')
            provider_options_list.append({})

    elif device_ep == 'DmlExecutionProvider':
        #print("Configuring DML Execution Provider...") # Отладочное сообщение
        dml_options = {
            'device_id': device_info.get_index(),
        }
        providers_list.append(device_ep)
        provider_options_list.append(dml_options)
        if 'CPUExecutionProvider' in rt.get_available_providers():
            providers_list.append('CPUExecutionProvider')
            provider_options_list.append({})
    else:
        # Для CPU или других провайдеров
        #print(f"Configuring {device_ep}...")
        providers_list.append(device_ep)
        provider_options_list.append({}) # Обычно CPU не требует опций


    #print(f"Creating InferenceSession with providers: {providers_list}")
    #print(f"Provider options: {provider_options_list}")

    # --- СОЗДАНИЕ СЕССИИ ---
    # Используем новый формат передачи опций
    sess = rt.InferenceSession(onnx_model_or_path,
                               sess_options=sess_options,
                               providers=providers_list,
                               provider_options=provider_options_list)

    #print("InferenceSession created successfully.")
    return sess