@echo off
echo Starting TASH Live Demo...
pushd "%~dp0"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo [ERROR] Virtual environment not found at %PY%
    echo Create it first:
    echo     py -3.12 -m venv .venv
    echo     %PY% -m pip install -e ".[live]"
    echo     %PY% -m tash.audio.download_model
    goto :end
)

"%PY%" -m tash.live

:end
popd
pause
