# AtualizaYolo
As Arquivos e repositórios que eu geralmente uso

Principalmente Ultralytics, Roboflow, e Supervision (recente)

Antes eu utilizava alguns dos algoritmos próprios do Yolo, os Weights, e treinados talvez eu coloque em uma pasta separada

Definir o ambiente com

```bash
python -m venv venv
venv/Scripts/Activate
# caso desejar sair
deactivate
```

e para instalar algumas das dependencias 
```bash
pip install ultralytics roboflow opencv-python torch numpy yt-dlp
```
Utiliza alguns outros repositórios como 
## 💻 Python

[**Python>=3.8**](https://www.python.org/) environment.

## 👀Supervison

[**Repositório do Git Time in Zone**](https://github.com/roboflow/supervision/tree/develop/examples/time_in_zone)

## 📅 Ultralytics

[**Repositório do Git Yolo**](https://github.com/ultralytics/ultralytics)

## 📺 Baixar Vídeos

[**FFmpeg**](https://github.com/GyanD/codexffmpeg/releases/tag/2025-05-26-git-43a69886b2)

## 🖼️ Label Studio
Uma ferramenta para auxiliar com os Labors, para o treinamento do modelo Yolo, pode ser integrado também com Yolo detect para ajudar com o auto-labor

e no caso utiliza Docker, que eu precisei fazer o WSL, e o Docker Desktop funcionar.

no arquivo eu preciso configurar coisas simples como colocar os modelos na pasta modelo, dps de dar Git Clone no Repositório

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend/tree/master

pip install label-studio

label-studio start
# no cd da pasta examples/yolo, dps de devidamente configurado com o Legacy Token

docker-compose build

docker-compose up
```
Toda informação necessária no próprio Repositório

[**Repositorio Git**](https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/examples/yolo/README.md)

