# Project

## Setup host system
```
chmod +x environment/base-packages.sh
sudo ./environment/base-packages.sh
```

## Setup and enter the docker image environment 

In linux

```
./docker.sh [clean]
```

## Setup and enter the virtual environment 

In windows

```
env.bat 
```


In linux

```
source ./env.sh [clean]
```

In docker

```
source ./env.sh [clean]
```

## Update docker environment

In the file (environment/base-packeges.sh) add apt packages that you need in your project

note that a newline will brake the RUN command and thus "\\" should be used when adding dependencies. More information on how docker works can be found on https://docs.docker.com/get-started/


## Update pip environment

the python dependencies for the project should be added to the pyproject.toml file

## Run tests using 

```
tox
```
