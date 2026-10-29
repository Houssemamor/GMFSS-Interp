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
    goto :venv
)

call :install_ffmpeg
if errorlevel 1 goto :ffmpeg_failed
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
"%PY%" -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo       FAILED: PyTorch installation did not complete
    pause
    exit /b 1
)

rem --------------------------------------------------------- 4. other deps
echo.
echo [4/5] Supporting packages (cupy + CUDA headers, about 1.5 GB)
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo       FAILED: could not install the packages in requirements.txt
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

rem ------------------------------------------------- subroutine: ffmpeg install
rem Kept outside any parenthesised block on purpose: %VAR% is expanded when a
rem whole block is parsed, so a variable set inside a block would read as empty.
:install_ffmpeg
echo       Downloading ffmpeg 7.0 ^(about 90 MB^)
if not exist "tools" mkdir "tools"
rem A version-pinned URL plus a SHA-256 check: the moving "release" URL cannot
rem be verified, and an unverified binary would run with this user's rights.
set "FFMPEG_SHA256=3333a744e1585375aef46f2055ea483d002de52bf8bf445638b80e1d86a2521e"
curl.exe -L --retry 3 --retry-delay 5 -o "tools\ffmpeg.zip" "https://github.com/GyanD/codexffmpeg/releases/download/7.0/ffmpeg-7.0-essentials_build.zip"
if errorlevel 1 exit /b 1
set "ZIPHASH="
for /f "skip=1 tokens=*" %%H in ('certutil -hashfile "tools\ffmpeg.zip" SHA256') do if not defined ZIPHASH set "ZIPHASH=%%H"
set "ZIPHASH=%ZIPHASH: =%"
if /i not "%ZIPHASH%"=="%FFMPEG_SHA256%" goto :install_ffmpeg_bad_hash
if not exist "tools\unpack" mkdir "tools\unpack"
tar -xf "tools\ffmpeg.zip" -C "tools\unpack"
if errorlevel 1 exit /b 1
rem dir /s lists only files that exist; "for /r" with a bare name would visit
rem every folder whether or not the file is there.
for /f "delims=" %%F in ('dir /s /b "tools\unpack\ffmpeg.exe"') do copy /y "%%F" "tools\ffmpeg.exe" >nul
for /f "delims=" %%F in ('dir /s /b "tools\unpack\ffprobe.exe"') do copy /y "%%F" "tools\ffprobe.exe" >nul
rmdir /s /q "tools\unpack"
del "tools\ffmpeg.zip"
if not exist "tools\ffmpeg.exe" exit /b 1
if not exist "tools\ffprobe.exe" exit /b 1
echo       Installed to tools\
exit /b 0

:install_ffmpeg_bad_hash
echo       SHA-256 mismatch for the ffmpeg download; discarding it.
del "tools\ffmpeg.zip"
exit /b 1
