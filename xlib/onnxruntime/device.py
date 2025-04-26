import ctypes
import itertools
import os
from typing import List, Tuple

import onnxruntime as rt

# Импорт appargs все еще может быть нужен для флага NO_CUDA
from .. import appargs as lib_appargs
# УДАЛЕНО: Логика получения флагов USE_TENSORRT_ONLY и USE_CUDA_ONLY здесь не нужна
# lib_appargs.get_arg_bool('USE_TENSORRT_ONLY')
# lib_appargs.get_arg_bool('USE_CUDA_ONLY')


class ORTDeviceInfo:
    """
    Represents picklable ONNXRuntime device info.
    Uniqueness is defined by (index, execution_provider).
    """
    def __init__(self, index=None, execution_provider=None, name=None, total_memory=None, free_memory=None):
        self._index : int = index # Physical device index (-1 for CPU)
        self._execution_provider : str = execution_provider
        self._name : str = name
        self._total_memory : int = total_memory
        self._free_memory : int = free_memory

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, d):
        self.__init__()
        self.__dict__.update(d)

    def is_cpu(self) -> bool: return self._index == -1

    def get_index(self) -> int:
        """Returns the physical device index."""
        return self._index

    def get_execution_provider(self) -> str:
        return self._execution_provider

    def get_name(self) -> str:
        return self._name

    def get_total_memory(self) -> int:
        return self._total_memory

    def get_free_memory(self) -> int:
        return self._free_memory

    def __eq__(self, other):
        # Уникальность определяется парой (индекс, провайдер)
        if isinstance(other, ORTDeviceInfo):
            return self._index == other._index and \
                   self._execution_provider == other._execution_provider
        return False

    def __hash__(self):
        # Хеш должен быть согласован с __eq__
        return hash((self._index, self._execution_provider))

    def __str__(self):
        if self.is_cpu():
            return f"CPU"
        else:
            ep = self.get_execution_provider()
            mem_gb_str = f"{(self._total_memory / 1024**3) :.3f}Gb"
            # Используем .startswith() для учета версий (например, TensorrtExecutionProvider_1_0)
            if ep.startswith('CUDAExecutionProvider'):
                ep_short = "CUDA"
            elif ep.startswith('TensorrtExecutionProvider'):
                 ep_short = "TensorRT"
            elif ep.startswith('DmlExecutionProvider'):
                 ep_short = "DirectX12"
            else:
                ep_short = ep # Fallback

            return f"[{self._index}] {self._name} [{mem_gb_str}] [{ep_short}]"

    def __repr__(self):
        return f'{self.__class__.__name__} object: ' + self.__str__()

# --- Глобальные переменные и функции ---

_ort_devices_info_cache: List[ORTDeviceInfo] = None # Кэш для get_available_devices_info

def get_cpu_device_info() -> ORTDeviceInfo:
    return ORTDeviceInfo(index=-1, execution_provider='CPUExecutionProvider', name='CPU', total_memory=0, free_memory=0)

def get_available_devices_info(include_cpu=True, cpu_only=False) -> List[ORTDeviceInfo]:
    """
    Returns a list of available ORTDeviceInfo based on environment variables
    set during the initial _initialize_ort_devices_info call.
    """
    global _ort_devices_info_cache
    if _ort_devices_info_cache is not None:
        # Возвращаем из кэша, если уже читали
        devices_to_return = _ort_devices_info_cache[:] # Копия
        if include_cpu and get_cpu_device_info() not in devices_to_return:
             devices_to_return.append(get_cpu_device_info())
        elif not include_cpu and get_cpu_device_info() in devices_to_return:
             devices_to_return.remove(get_cpu_device_info())

        # Применяем cpu_only к кэшированным данным
        if cpu_only:
            return [d for d in devices_to_return if d.is_cpu()]
        else:
            return devices_to_return

    # Читаем из переменных окружения первый раз
    detected_devices = []
    if not cpu_only:
        count = int(os.environ.get('ORT_DEVICES_COUNT', 0))
        #print(f"get_available_devices_info: Found {count} devices in environment.") # Отладка
        for i in range(count):
            try:
                device_info = ORTDeviceInfo(
                    index=int(os.environ[f'ORT_DEVICE_{i}_PHYSICAL_INDEX']), # Используем новый ключ
                    execution_provider=os.environ[f'ORT_DEVICE_{i}_EP'],
                    name=os.environ[f'ORT_DEVICE_{i}_NAME'],
                    total_memory=int(os.environ[f'ORT_DEVICE_{i}_TOTAL_MEM']),
                    free_memory=int(os.environ[f'ORT_DEVICE_{i}_FREE_MEM']),
                )
                detected_devices.append(device_info)
                #print(f"get_available_devices_info: Loaded device {i}: {device_info}") # Отладка
            except KeyError as e:
                print(f"Warning: Environment variables for ORT device {i} not fully set (missing {e}). Skipping.")
                continue
            except ValueError as e:
                print(f"Warning: Invalid value in environment variables for ORT device {i} ({e}). Skipping.")
                continue

    _ort_devices_info_cache = detected_devices[:] # Сохраняем в кэш (без CPU)

    # Добавляем CPU если нужно
    if include_cpu:
        cpu_dev = get_cpu_device_info()
        if cpu_dev not in detected_devices: # Проверяем, чтобы не дублировать, если вдруг он там оказался
            detected_devices.append(cpu_dev)

    return detected_devices


def _initialize_ort_devices_info():
    """
    Determines available ORT devices based on providers and arguments,
    then stores info about them in os.environ for potential use in subprocesses.
    This function runs only once per process on module import.
    A single physical GPU can result in multiple entries (CUDA, TensorRT).
    """
    if int(os.environ.get('ORT_DEVICES_INITIALIZED', 0)) == 1:
        # Уже инициализировано в этом процессе
        #print("_initialize_ort_devices_info: Already initialized.")
        return

    os.environ['ORT_DEVICES_INITIALIZED'] = '1'
    os.environ['ORT_DEVICES_COUNT'] = '0'
    #print("_initialize_ort_devices_info: Initializing...")

    devices = [] # Список словарей для найденных устройств
    try:
        prs = rt.get_available_providers()
        #print(f"ONNX Runtime Available Providers: {prs}")
    except Exception as e:
        print(f"Warning: Could not get available ONNX Runtime providers: {e}")
        prs = []

    # --- Проверка аргументов командной строки (только NO_CUDA) ---
    no_cuda = lib_appargs.get_arg_bool('NO_CUDA')
    # УДАЛЕНО: Получение флагов USE_TENSORRT_ONLY и USE_CUDA_ONLY
    # use_tensorrt_only = lib_appargs.get_arg_bool('USE_TENSORRT_ONLY')
    # use_cuda_only = lib_appargs.get_arg_bool('USE_CUDA_ONLY')

    # УДАЛЕНО: Проверка конфликта флагов
    # if use_tensorrt_only and use_cuda_only:
    #    print("Warning: Both USE_TENSORRT_ONLY and USE_CUDA_ONLY are set. Prioritizing USE_TENSORRT_ONLY.")
    #    use_cuda_only = False # Отдаем приоритет TRT

    # --- Определение доступности провайдеров ---
    can_use_cuda_ep = 'CUDAExecutionProvider' in prs
    can_use_tensorrt_ep = 'TensorrtExecutionProvider' in prs
    can_use_dml_ep = 'DmlExecutionProvider' in prs

    # --- Поиск устройств NVIDIA (CUDA/TensorRT) ---
    should_check_nvidia = (can_use_cuda_ep or can_use_tensorrt_ep) and not no_cuda

    if should_check_nvidia:
        #print("Checking for NVIDIA devices...")
        os.environ['CUDA_CACHE_MAXSIZE'] = '2147483647'
        try:
            libnames = ('libcuda.so', 'libcuda.dylib', 'nvcuda.dll')
            cuda = None
            for libname in libnames:
                try:
                    # Добавляем winmode=0 для Windows, чтобы избежать поиска в System32 перед PATH
                    load_kw = {'winmode': 0} if os.name == 'nt' else {}
                    cuda = ctypes.CDLL(libname, **load_kw)
                    #print(f"Loaded CUDA library: {libname}")
                    break
                except OSError:
                    continue

            if cuda is None:
                print("Could not load CUDA driver library. Skipping NVIDIA device detection.")
            else:
                 # Проверка наличия функций (добавлена для надежности)
                required_funcs = ['cuInit', 'cuDeviceGetCount', 'cuDeviceGet', 'cuDeviceGetName',
                                  'cuDeviceComputeCapability', 'cuCtxCreate_v2', 'cuMemGetInfo_v2', 'cuCtxDetach']
                if not all(hasattr(cuda, f) for f in required_funcs):
                     print(f"CUDA library loaded, but missing required functions. Skipping NVIDIA detection.")
                else:
                    # Задаем типы аргументов и возвращаемых значений
                    cuda.cuInit.argtypes = [ctypes.c_uint]
                    cuda.cuInit.restype = ctypes.c_int
                    cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
                    cuda.cuDeviceGetCount.restype = ctypes.c_int
                    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
                    cuda.cuDeviceGet.restype = ctypes.c_int
                    cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
                    cuda.cuDeviceGetName.restype = ctypes.c_int
                    cuda.cuDeviceComputeCapability.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
                    cuda.cuDeviceComputeCapability.restype = ctypes.c_int
                    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
                    cuda.cuCtxCreate_v2.restype = ctypes.c_int
                    cuda.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
                    cuda.cuMemGetInfo_v2.restype = ctypes.c_int
                    cuda.cuCtxDetach.argtypes = [ctypes.c_void_p]
                    cuda.cuCtxDetach.restype = ctypes.c_int

                    nGpus = ctypes.c_int()
                    name_buffer = ctypes.create_string_buffer(200)
                    cc_major = ctypes.c_int()
                    cc_minor = ctypes.c_int()
                    freeMem = ctypes.c_size_t()
                    totalMem = ctypes.c_size_t()
                    device_handle = ctypes.c_int() # Изменено имя для ясности
                    context = ctypes.c_void_p()

                    res_init = cuda.cuInit(0)
                    if res_init == 0: # CUDA_SUCCESS
                        #print("cuInit successful.")
                        res_count = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
                        if res_count == 0 and nGpus.value > 0:
                            #print(f"cuDeviceGetCount found {nGpus.value} GPUs.")
                            for physical_idx in range(nGpus.value):
                                # Получаем информацию о ФИЗИЧЕСКОМ устройстве
                                if cuda.cuDeviceGet(ctypes.byref(device_handle), physical_idx) != 0: continue
                                if cuda.cuDeviceGetName(name_buffer, ctypes.sizeof(name_buffer), device_handle.value) != 0: continue
                                if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device_handle.value) != 0: continue

                                device_name = name_buffer.value.decode('utf-8')
                                device_total_mem = 0
                                device_free_mem = 0

                                # Пытаемся получить память
                                context.value = None
                                if cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device_handle.value) == 0:
                                    if cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem)) == 0:
                                        device_total_mem = totalMem.value
                                        device_free_mem = freeMem.value
                                    else:
                                         print(f"Warning: cuMemGetInfo_v2 failed for device {physical_idx}, memory info unavailable.")
                                    cuda.cuCtxDetach(context) # Отсоединяем контекст
                                else:
                                     print(f"Warning: cuCtxCreate_v2 failed for device {physical_idx}, memory info unavailable.")

                                # Теперь решаем, какие Execution Provider добавить для этой карты
                                added_cuda = False
                                # ИЗМЕНЕНО: Убрана проверка 'and not use_tensorrt_only'
                                if can_use_cuda_ep:
                                    devices.append({
                                        'physical_index': physical_idx, # Сохраняем физический индекс
                                        'execution_provider': 'CUDAExecutionProvider',
                                        'name': device_name,
                                        'total_mem': device_total_mem,
                                        'free_mem': device_free_mem,
                                    })
                                    added_cuda = True
                                    #print(f"  Added device: [{physical_idx}] {device_name} as CUDAExecutionProvider")

                                added_trt = False
                                # ИЗМЕНЕНО: Убрана проверка 'and not use_cuda_only'
                                if can_use_tensorrt_ep:
                                    devices.append({
                                        'physical_index': physical_idx, # Тот же физический индекс
                                        'execution_provider': 'TensorrtExecutionProvider',
                                        'name': device_name, # То же имя
                                        'total_mem': device_total_mem, # Та же память
                                        'free_mem': device_free_mem, # Та же память
                                    })
                                    added_trt = True
                                    #print(f"  Added device: [{physical_idx}] {device_name} as TensorrtExecutionProvider")

                                if not added_cuda and not added_trt:
                                    # ИЗМЕНЕНО: Обновлено сообщение об пропуске
                                    print(f"  Skipped device [{physical_idx}] {device_name} (CUDA/TensorRT providers not available/enabled).")

                        elif res_count != 0:
                             print(f"cuDeviceGetCount failed with error code: {res_count}")
                        else: # nGpus.value == 0
                             print("cuDeviceGetCount reported 0 GPUs.")
                    else:
                        print(f"cuInit failed with error code: {res_init}. Cannot query NVIDIA devices.")
        except Exception as e:
            import traceback
            print(f'Error during NVIDIA device detection: {e}')
            print(traceback.format_exc())
    elif no_cuda:
        print("NVIDIA device check skipped due to NO_CUDA flag.")
    else:
        print("NVIDIA device check skipped (CUDA/TensorRT providers not found in ONNX Runtime).")


    # --- Поиск устройств DirectML ---
    if can_use_dml_ep:
        #print("Checking for DirectML devices...")
        try:
            # Убедитесь, что xlib установлен и доступен: pip install xlib
            # или предоставьте свой способ получения информации DXGI
            from xlib.api.win32 import dxgi as lib_dxgi # Убедитесь, что импорт работает
            dxgi_factory = lib_dxgi.create_DXGIFactory4()
            if dxgi_factory is not None:
                #print("DXGIFactory4 created.")
                for adapter_idx in itertools.count(): # Индекс адаптера DXGI
                    adapter = None
                    try:
                        adapter = dxgi_factory.enum_adapters1(adapter_idx)
                        if adapter is None:
                            break # Адаптеры закончились

                        desc = adapter.get_desc1()
                        is_software = (desc.Flags & lib_dxgi.DXGI_ADAPTER_FLAG.DXGI_ADAPTER_FLAG_SOFTWARE) != 0
                        is_remote = (desc.VendorId == 0x1414 and desc.DeviceId == 0x8c) # MS Basic Render / Remote

                        if not is_software and not is_remote:
                            devices.append({
                                'physical_index': adapter_idx, # Используем индекс адаптера как "физический"
                                'execution_provider': 'DmlExecutionProvider',
                                'name': desc.Description,
                                'total_mem': desc.DedicatedVideoMemory,
                                'free_mem': desc.DedicatedVideoMemory, # DXGI не дает свободной памяти
                            })
                            #print(f"  Added device: [{adapter_idx}] {desc.Description} as DmlExecutionProvider")
                        else:
                             print(f"  Skipped DXGI adapter [{adapter_idx}] {desc.Description} (Software/Remote).")

                    except Exception as adapter_error:
                        print(f"Error processing DXGI adapter {adapter_idx}: {adapter_error}. Stopping enumeration.")
                        break # Прерываем при ошибке
                    finally:
                        if adapter: adapter.Release()
                dxgi_factory.Release()
            else:
                print("Failed to create DXGIFactory4.")
        except ImportError:
            print("Could not import 'xlib.api.win32.dxgi'. Skipping DirectML detection.")
        except Exception as e:
            import traceback
            print(f'Error during DirectML device detection: {e}')
            print(traceback.format_exc())
    else:
        pass
        #print("DirectML device check skipped (DmlExecutionProvider not found in ONNX Runtime).")


    # --- Запись найденных устройств в переменные окружения ---
    # Используем уникальный порядковый номер для ключей env var (0, 1, 2...)
    # но сохраняем исходный физический индекс внутри значения
    #print(f"Registering {len(devices)} logical devices in environment variables...")
    os.environ['ORT_DEVICES_COUNT'] = str(len(devices))
    for i, device_dict in enumerate(devices):
        #print(f"  Registering env ORT_DEVICE_{i}: {device_dict}")
        os.environ[f'ORT_DEVICE_{i}_PHYSICAL_INDEX'] = str(device_dict['physical_index'])
        os.environ[f'ORT_DEVICE_{i}_EP'] = device_dict['execution_provider']
        os.environ[f'ORT_DEVICE_{i}_NAME'] = device_dict['name']
        os.environ[f'ORT_DEVICE_{i}_TOTAL_MEM'] = str(device_dict['total_mem'])
        os.environ[f'ORT_DEVICE_{i}_FREE_MEM'] = str(device_dict['free_mem'])

    #print("_initialize_ort_devices_info: Initialization complete.")

# ======================================================
# Вызываем инициализацию при импорте модуля ОДИН РАЗ
# ======================================================
_initialize_ort_devices_info()
