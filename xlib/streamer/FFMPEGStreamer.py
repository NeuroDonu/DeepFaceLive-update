# xlib\streamer\FFMPEGStreamer.py
import numpy as np
import subprocess
import sys
from typing import Optional

# Пытаемся импортировать torch для проверки CUDA
_gpu_available = False
try:
    import torch
    if torch.cuda.is_available():
        _gpu_available = True
        # print("FFMPEGStreamer: NVIDIA GPU detected via torch.cuda.") # Убрано
    # else: # Убрано
        # print("FFMPEGStreamer: torch found, but torch.cuda.is_available() is False.") # Убрано
except ImportError:
    # print("FFMPEGStreamer: torch not found. GPU acceleration (NVENC) will not be used.") # Убрано
    pass
except Exception as e:
    # Оставляем вывод ошибки при проверке CUDA
    print(f"FFMPEGStreamer: Error during torch/cuda check: {e}. GPU acceleration not used.")


from .. import ffmpeg as lib_ffmpeg

class FFMPEGStreamer:
    def __init__(self):
        self._ffmpeg_proc = None
        self._addr = '127.0.0.1'
        self._port = 1234
        self._width = 320
        self._height = 240

        # --- Параметры кодирования по умолчанию ---
        if _gpu_available:
            # Параметры для NVIDIA NVENC
            self._codec = 'h264_nvenc'
            self._preset = 'p1' # Самый быстрый
            self._tune = 'll' # Low Latency
            self._use_crf = False
            self._bitrate = '4000k'
            self._max_bitrate = '6000k'
            self._bufsize = '8000k'
            # print("FFMPEGStreamer: Using NVENC hardware encoding.") # Убрано
        else:
            # Параметры для CPU libx264
            self._codec = 'libx264'
            self._preset = 'ultrafast'
            self._tune = 'zerolatency'
            self._use_crf = True
            self._crf = '28'
            # print("FFMPEGStreamer: Using libx264 CPU encoding.") # Убрано

        # Общие параметры
        self._gop_size = 120
        self._framerate = 60
        self._pix_fmt_out = 'yuv420p'
        # --- ---

    def set_addr_port(self, addr : str, port : int):
        """ Устанавливает адрес и порт назначения UDP. """
        if self._addr != addr or self._port != port:
            # print(f"FFMPEGStreamer: UDP target changed to {addr}:{port}. Restarting...") # Убрано
            self._addr = addr
            self._port = port
            self.stop() # Останавливаем старый процесс, если параметры изменились

    def stop(self):
        """ Останавливает процесс FFMPEG, если он запущен. """
        proc = self._ffmpeg_proc
        if proc is not None:
            # print("FFMPEGStreamer: Stopping FFMPEG process...") # Убрано
            self._ffmpeg_proc = None
            if proc.stdin:
                try: proc.stdin.close()
                except Exception: pass
            if proc.stderr:
                try: proc.stderr.close()
                except Exception: pass

            try:
                proc.terminate()
                try:
                    proc.wait(timeout=0.5)
                    # print("FFMPEGStreamer: Process terminated gracefully.") # Убрано
                except subprocess.TimeoutExpired:
                    # Оставляем вывод о проблемах с остановкой
                    print("FFMPEGStreamer: Process did not terminate gracefully, killing.")
                    proc.kill()
                    try: proc.wait(timeout=0.5)
                    except Exception: pass # Игнорируем ошибки ожидания после kill
                    # print("FFMPEGStreamer: Process killed.") # Убрано
                except Exception as e:
                    print(f"FFMPEGStreamer: Exception during process wait/kill: {e}") # Оставляем

            except Exception as e:
                 print(f"FFMPEGStreamer: Exception during process terminate: {e}") # Оставляем
                 try:
                     proc.kill()
                     try: proc.wait(timeout=0.5)
                     except Exception: pass
                     # print("FFMPEGStreamer: Process killed after terminate exception.") # Убрано
                 except Exception as ke:
                     print(f"FFMPEGStreamer: Exception during final kill attempt: {ke}") # Оставляем
        # else:
            # print("FFMPEGStreamer: Stop called, but process was not running.") # Убрано


    def _restart(self):
        """ Перезапускает процесс FFMPEG с текущими параметрами. """
        self.stop() # Гарантированно останавливаем предыдущий процесс

        if not self._addr or not self._port:
            # Оставляем вывод об ошибке конфигурации
            print("FFMPEGStreamer: Cannot restart - UDP address or port is not set.")
            return

        # print(f"FFMPEGStreamer: Restarting for UDP {self._addr}:{self._port}...") # Убрано

        # --- Формирование аргументов FFMPEG ---
        input_args = ['-y', '-f', 'rawvideo', '-vcodec','rawvideo', '-pix_fmt', 'bgr24',
                      '-s', f'{self._width}x{self._height}', '-r', str(self._framerate), '-i', '-']

        codec_args = ['-c:v', self._codec, '-preset', self._preset, '-tune', self._tune,
                      '-pix_fmt', self._pix_fmt_out, '-r', str(self._framerate), '-g', str(self._gop_size)]
        if self._use_crf:
            codec_args.extend(['-crf', self._crf])
        else:
            codec_args.extend(['-b:v', self._bitrate, '-maxrate', self._max_bitrate, '-bufsize', self._bufsize])

        audio_args = ['-an']

        output_args = ['-f', 'mpegts', '-mpegts_flags', '+initial_discontinuity', '-flush_packets', '1',
                       f'udp://{self._addr}:{self._port}?pkt_size=1316&buffer_size=65535&fifo_size=1000000']

        args = input_args + codec_args + audio_args + output_args
        # --- Конец формирования аргументов ---

        # print(f"FFMPEGStreamer: Starting FFMPEG with args: {' '.join(args)}") # Убрано
        try:
            self._ffmpeg_proc = lib_ffmpeg.run (args, pipe_stdin=True, quiet_stderr=True, pipe_stderr=False)
            # print("FFMPEGStreamer: FFMPEG process started successfully.") # Убрано
        except Exception as e:
            # Оставляем вывод об ошибке запуска
            print(f"!!! FFMPEGStreamer: Failed to start FFMPEG: {e}")
            self._ffmpeg_proc = None


    def push_frame(self, img : np.ndarray):
        """ Отправляет кадр в FFMPEG. """
        if img is None or img.ndim != 3 or img.dtype != np.uint8:
            # Оставляем вывод о невалидном кадре
            print(f"FFMPEGStreamer: Invalid frame received. Got shape={img.shape if img is not None else 'None'}, dtype={img.dtype if img is not None else 'N/A'}")
            return

        H,W,C = img.shape
        if C != 3:
            # Оставляем вывод о неверном количестве каналов
            print(f"FFMPEGStreamer: Expected 3 channels (BGR), got {C}. Frame skipped.")
            return

        # Проверка и перезапуск при изменении размера
        if self._width != W or self._height != H:
            # Оставляем вывод о смене размера
            print(f"FFMPEGStreamer: Frame dimensions changed ({W}x{H}). Restarting...")
            self._width = W
            self._height = H
            self.stop() # Останавливаем старый процесс

        # Запускаем, если еще не запущен (или был остановлен)
        if self._ffmpeg_proc is None:
            self._restart()

        # Если процесс все еще не запущен (ошибка при рестарте), выходим
        if self._ffmpeg_proc is None:
            return

        # Пишем кадр в stdin
        try:
            self._ffmpeg_proc.stdin.write(img.tobytes())
            self._ffmpeg_proc.stdin.flush() # Проталкиваем буфер
        except BrokenPipeError:
            # Оставляем вывод об ошибке Broken pipe
            print("FFMPEGStreamer: Broken pipe. FFMPEG process likely terminated. Stopping.")
            self.stop()
        except OSError as e:
             # Оставляем вывод об ошибке OSError
             if self._ffmpeg_proc and self._ffmpeg_proc.poll() is not None:
                 print(f"FFMPEGStreamer: FFMPEG process terminated unexpectedly (poll result: {self._ffmpeg_proc.poll()}). Stopping.")
             else:
                 print(f"FFMPEGStreamer: OSError writing to FFMPEG: {e}. Stopping.")
             self.stop()
        except Exception as e:
            # Оставляем вывод о других ошибках записи
            print(f"FFMPEGStreamer: Unexpected error writing frame: {type(e).__name__}: {e}. Stopping.")
            self.stop()