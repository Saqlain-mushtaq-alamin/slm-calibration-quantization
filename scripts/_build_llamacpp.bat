@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CMAKE_GENERATOR=Ninja
set CMAKE_BUILD_PARALLEL_LEVEL=4
D:\canvas\slm-calibration-quantization\.venv\Scripts\python.exe -m pip install "D:\canvas\slm-calibration-quantization\outputs\quantized\llama_cpp_src\extracted\llama_cpp_python-0.3.4" --no-build-isolation --timeout 300 --retries 2
