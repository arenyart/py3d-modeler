@echo off
setlocal
cd /d "%~dp0"

echo === Py3D Modeler: Building Single-File Windows EXE ===

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Installing dependencies and PyInstaller...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt pyinstaller

echo Packing dist\Py3DModeler.exe...
.venv\Scripts\pyinstaller.exe --clean --noconfirm Py3DModeler.spec

if exist "dist\Py3DModeler.exe" (
    echo.
    echo SUCCESS! Standalone executable built at:
    echo   %cd%\dist\Py3DModeler.exe
) else (
    echo.
    echo Build failed. Check the output above for errors.
    exit /b 1
)
pause
