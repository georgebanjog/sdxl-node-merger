@echo off
title SDXL Node Merger

echo ============================================================
echo    SDXL Node Merger - Launcher
echo ============================================================
echo.

cd /d "%~dp0"
echo [INFO] Working directory: %CD%
echo.

:: --- Check Python ---
echo [CHECK] Looking for Python...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH!
    echo         Install Python 3.10+ from https://python.org
    echo.
    goto :error_exit
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo [OK] Found: %%i
echo.

:: --- Handle existing broken venv ---
if exist ".venv" if not exist ".venv\Scripts\python.exe" (
    echo [WARNING] Removing broken .venv...
    rmdir /s /q ".venv" 2>nul
)

:: --- Create virtual environment ---
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 goto :error_exit
    echo [OK] Virtual environment created.
    echo.
)

:: --- Activate venv ---
echo [INFO] Activating virtual environment...
call ".venv\Scripts\activate.bat"
echo [OK] Activated.
echo.

:: --- Install dependencies ---
if exist ".venv\.deps_installed" goto :deps_done

echo ============================================================
echo    Installing dependencies - first time only
echo ============================================================
echo.

echo [1/3] Upgrading pip...
python -m pip install --upgrade pip
echo.

:: --- Try PyTorch CUDA versions one by one ---
echo [2/3] Installing PyTorch...

echo        Trying CUDA 12.8...
pip install torch --index-url https://download.pytorch.org/whl/cu128 2>nul
if %errorlevel% equ 0 goto :torch_done

echo        Trying CUDA 12.6...
pip install torch --index-url https://download.pytorch.org/whl/cu126 2>nul
if %errorlevel% equ 0 goto :torch_done

echo        Trying CUDA 12.4...
pip install torch --index-url https://download.pytorch.org/whl/cu124 2>nul
if %errorlevel% equ 0 goto :torch_done

echo        All CUDA failed. Installing CPU version...
pip install torch
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch!
    goto :error_exit
)

:torch_done
echo [OK] PyTorch installed.
echo.

:: Show CUDA status
python -c "import torch; print('    PyTorch', torch.__version__); print('    CUDA:', torch.cuda.is_available())"
echo.

echo [3/3] Installing safetensors, websockets and extras...
pip install safetensors websockets packaging numpy
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install safetensors or websockets!
    goto :error_exit
)
echo.

echo done > ".venv\.deps_installed"
echo [OK] All dependencies installed!
echo.

:deps_done
echo [OK] Dependencies ready.
echo.

:: --- Create projects dir ---
if not exist "projects" mkdir projects

:: --- Launch server ---
echo ============================================================
echo    Starting SDXL Node Merger...
echo    URL: http://127.0.0.1:8765
echo    Press Ctrl+C to stop.
echo ============================================================
echo.

python server.py

echo.
echo [INFO] Server stopped.
goto :end

:error_exit
echo.
echo [ERROR] Something went wrong. See above.

:end
echo.
pause
