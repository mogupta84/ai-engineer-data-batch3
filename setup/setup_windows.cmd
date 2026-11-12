@echo off
REM AI Engineer (Data), Batch 3, Days 1 to 4: one-time setup on Windows.
REM Run it from Command Prompt, inside the repository folder:   setup\setup_windows.cmd
REM Safe to run again; it skips what is already done.
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1

echo.
echo [1/4] Checking Python (3.11 or newer)...
py -3 -c "import sys; assert sys.version_info >= (3, 11), sys.version; print('Python', sys.version.split()[0])"
if errorlevel 1 (
  echo Python 3.11 or newer was not found. Install it from python.org first, then run this again.
  exit /b 1
)

echo.
echo [2/4] Creating the virtual environment in .venv ...
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  if errorlevel 1 exit /b 1
)
call .venv\Scripts\activate.bat

echo.
echo [3/4] Installing packages. This takes 5 to 10 minutes the first time...
python -m pip install --upgrade pip --quiet --disable-pip-version-check
python -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 exit /b 1

echo.
echo [4/4] Your settings file...
if not exist .env (
  copy .env.example .env >nul
  echo Made .env from .env.example. Open it and fill in the lines your trainer gives you.
) else (
  echo .env already exists; left as it is.
)
python lab1_foundry_prompts\l1_tokens.py
if errorlevel 1 exit /b 1

echo.
echo Setup finished. Next: az login, then follow the Day 1 handbook.
endlocal
