@echo off
rem Double-click this file to start the dashboard (Windows).
cd /d "%~dp0"
set PY=python
where python >nul 2>&1 || set PY=py
%PY% run.py check >nul 2>&1
if errorlevel 1 (
  echo Installing the needed packages. This happens only the first time...
  %PY% -m pip install -r requirements.txt
)
%PY% run.py
pause
