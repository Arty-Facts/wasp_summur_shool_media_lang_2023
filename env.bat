echo off
set clean=%1
if /I "%clean%" EQU "clean" rmdir /s venv 
echo on

@REM create venv if it dose not exists
if not exist "venv" python -m venv venv

if not exist "tortoise-tts" git clone git clone https://github.com/neonbjb/tortoise-tts.git

cd tortoise-tts 
venv\Scripts\pip install .
cd ..

venv\Scripts\pip install -r tortoise-tts/requirements.txt

@REM assume that windows systems has a nvidia GPU
venv\Scripts\pip install -e .

@REM start env
venv\Scripts\Activate