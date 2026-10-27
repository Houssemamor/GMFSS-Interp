@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if "%~1"=="" (
    echo Drag one or more video files onto this file, or run from a prompt:
    echo     Interpolate.bat "D:\video\episode.mkv"
    echo.
    echo Settings live in config.toml next to this file.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo No environment found. Run setup.bat once before interpolating.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m src.main --config "config.toml" %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" goto :finished
echo Finished with errors ^(exit code %RC%^).
pause
exit /b 1

:finished
echo Finished.
pause
exit /b 0
