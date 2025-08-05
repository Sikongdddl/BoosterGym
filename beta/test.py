import ctypes
try:
    ctypes.CDLL('libcuda.so')
    print("libcuda.so loaded successfully")
except OSError as e:
    print("Failed to load libcuda.so:", e)