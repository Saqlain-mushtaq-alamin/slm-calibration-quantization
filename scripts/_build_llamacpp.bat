@echo off
REM Build llama-cpp-python with CUDA GPU support
REM If building from source with MSVC and CUDA Toolkit:
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CMAKE_GENERATOR=Ninja
set CMAKE_BUILD_PARALLEL_LEVEL=4
set CMAKE_ARGS=-DGGML_CUDA=on
set FORCE_CMAKE=1

REM Install source build:
D:\canvas\slm-calibration-quantization\.venv\Scripts\python.exe -m pip install "D:\canvas\slm-calibration-quantization\outputs\quantized\llama_cpp_src\extracted\llama_cpp_python-0.3.4" --no-build-isolation --timeout 300 --retries 2

REM Note: If building from source without CUDA Toolkit installed, install pre-built CUDA 12.1 wheel:
REM python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --force-reinstall --no-cache-dir
