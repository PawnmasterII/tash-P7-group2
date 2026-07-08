@echo off
echo Starting TASH Live Demo...
pushd "%~dp0.."
py -3.12 -m tash.live
popd
pause
