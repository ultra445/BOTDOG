@echo off
setlocal
cd /d %~dp0
call .\.venv\Scripts\activate.bat
if not exist logs mkdir logs
set PYTHONUNBUFFERED=1
python -m src.dogbot.run >> logs\dogbot.out.log 2>> logs\dogbot.err.log
