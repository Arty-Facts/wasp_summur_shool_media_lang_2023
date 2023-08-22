echo off
set clean=%1
if /I "%clean%" EQU "clean" rmdir /s venv 
echo on

@REM create venv if it dose not exists
if not exist "venv" python -m venv venv

if not exist "tortoise-tts" git clone https://github.com/neonbjb/tortoise-tts.git

venv\Scripts\pip install -r tortoise-tts/requirements.txt

cd tortoise-tts 
..\venv\Scripts\pip install .
cd ..


@REM assume that windows systems has a nvidia GPU
venv\Scripts\pip install -e .

@REM start env
venv\Scripts\Activate