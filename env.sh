#! /bin/bash
__banner()
{
    echo "============================================================"
    echo $*
    echo "============================================================"
}

if [[ $1 == "clean" ]]; then 
    __banner Remove old env...
    if [[ -d "venv" ]]; then 
        rm -rf venv/ 
    fi
fi

python3 -m venv venv

if [[ ! -d "tortoise-tts" ]]; then 
    __banner Cloning tortoise-tts...
    git clone https://github.com/neonbjb/tortoise-tts.git
fi
__banner Installing tortoise-tts...

cd tortoise-tts; ./venv/bin/pip install .; cd ..

./venv/bin/pip install -r tortoise-tts/requirements.txt

__banner Instaling base packages... 
./venv/bin/pip install -e .


chmod +x venv/bin/activate

# start env
source ./venv/bin/activate