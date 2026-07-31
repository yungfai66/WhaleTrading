@echo off
REM Launches WhaleTrading locally and opens it in your default browser.
REM Double-click this file, or run it from a terminal.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app.py
pause
