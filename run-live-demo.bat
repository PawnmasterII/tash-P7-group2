@echo off
echo Starting TASH Live Demo...
pushd "%~dp0"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo [ERROR] Virtual environment not found at %PY%
    echo Create it first ^(run from parent folder — repo types.py shadows stdlib^):
    echo     cd ..
    echo     py -3.12 -m venv "tash-P7-group2\.venv"
    echo     tash-P7-group2\.venv\Scripts\python.exe -m pip install -e ".\tash-P7-group2[live]"
    echo     tash-P7-group2\.venv\Scripts\python.exe -m tash.audio.download_model
    goto :end
)

"%PY%" -m tash.live

:end
popd
pause
