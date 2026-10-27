@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo  GMFSS-Interp setup
echo ============================================================
echo.

if not exist "config.toml" (
    echo config.toml is missing. Restore it from the project archive.
    pause
    exit /b 1
)

rem -------------------------------------------------------------- 1. ffmpeg
echo [1/5] ffmpeg
set "HAVE_FFMPEG="
where ffmpeg >nul 2>nul && set "HAVE_FFMPEG=1"
if exist "tools\ffmpeg.exe" set "HAVE_FFMPEG=1"
if defined HAVE_FFMPEG (
    echo       ffmpeg already available
) else (
    echo       Downloading a static ffmpeg build ^(about 110 MB^)
    if not exist "tools" mkdir "tools"
    curl.exe -L --retry 3 --retry-delay 5 -o "tools\ffmpeg.zip" ^
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    if errorlevel 1 goto :ffmpeg_failed
    if not exist "tools\unpack" mkdir "tools\unpack"
    tar -xf "tools\ffmpeg.zip" -C "tools\unpack"
    if errorlevel 1 goto :ffmpeg_failed
    for /r "tools\unpack" %%F in (ffmpeg.exe) do copy /y "%%F" "tools\ffmpeg.exe" >nul
    for /r "tools\unpack" %%F in (ffprobe.exe) do copy /y "%%F" "tools\ffprobe.exe" >nul
    rmdir /s /q "tools\unpack"
    del "tools\ffmpeg.zip"
    if not exist "tools\ffmpeg.exe" goto :ffmpeg_failed
    if not exist "tools\ffprobe.exe" goto :ffmpeg_failed
    echo       Installed to tools\
)
goto :venv

:ffmpeg_failed
echo       FAILED: could not fetch ffmpeg.
echo       Install it manually, add it to PATH, or set [paths] ffmpeg
echo       in config.toml to the full path of ffmpeg.exe.
pause
exit /b 1

rem ---------------------------------------------------------------- 2. venv
:venv
echo.
echo [2/5] Python environment
if exist ".venv\Scripts\python.exe" (
    echo       Reusing .venv
) else (
    echo       Creating .venv ^(Python 3.12^)
    where uv >nul 2>nul
    if not errorlevel 1 (
        uv venv --python 3.12 ".venv"
    ) else (
        py -3.12 -m venv ".venv" 2>nul
        if not exist ".venv\Scripts\python.exe" (
            python -m venv ".venv"
        )
    )
)
if not exist ".venv\Scripts\python.exe" (
    echo       FAILED: could not create .venv
    pause
    exit /b 1
)
set "PY=.venv\Scripts\python.exe"

rem uv-created environments ship without pip, so install it before anything else.
"%PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
    "%PY%" -m ensurepip --upgrade >nul 2>nul
)
"%PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo       FAILED: pip is not available in .venv
    pause
    exit /b 1
)

rem ------------------------------------------------------------- 3. pytorch
echo.
echo [3/5] PyTorch with CUDA 12.6 (about 2.4 GB on first install)
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo       FAILED: PyTorch installation did not complete
    pause
    exit /b 1
)

rem --------------------------------------------------------- 4. other deps
echo.
echo [4/5] Supporting packages (cupy + CUDA headers, about 1.5 GB)
"%PY%" -m pip install numpy "cupy-cuda12x[ctk]"
if errorlevel 1 (
    echo       FAILED: could not install numpy / cupy
    pause
    exit /b 1
)

rem --------------------------------------------------------------- 5. models
echo.
echo [5/5] Model weights
"%PY%" "src\download_models.py"
if errorlevel 1 (
    echo       Some weights are missing. Re-run setup.bat to retry.
    pause
    exit /b 1
)

rem -------------------------------------------------------------- self check
echo.
echo Verifying the GPU stack...
"%PY%" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no device')"
if errorlevel 1 (
    echo       FAILED: torch cannot see the GPU
    pause
    exit /b 1
)
"%PY%" -c "import cupy; p = cupy.cuda.runtime.getDeviceProperties(0); print('cupy', cupy.__version__, '|', p['name'].decode())"
if errorlevel 1 (
    echo       FAILED: cupy cannot reach the CUDA runtime.
    echo       Update the NVIDIA driver, then re-run setup.bat.
    echo       If it still fails, install the CUDA headers manually:
    echo         .venv\Scripts\python.exe -m pip install "cupy-cuda12x[ctk]"
    pause
    exit /b 1
)

echo.
echo Setup complete. Drag a video file onto Interpolate.bat.
pause
exit /b 0
