@echo off
REM EdgeMind CX Web - Sunucuyu 127.0.0.1:8001 adresinde baslatir.
REM Tarayicida acin: http://127.0.0.1:8001
cd /d "%~dp0"
python -m uvicorn web.main:app --host 127.0.0.1 --port 8001
pause
