# Classification REST API 만들기

## Setup
``` bash
conda create -n flask python=3.11
conda activate flask
pip install Flask Pillow torch torchvision torchaudio
```

## Start Demo
```bash
python {Python Script name}
#ex) python main.py
```

## How to Use REST API
```bash
curl -XGET {ADDRESS:PORT}/ping
# ex) curl -XGET 0.0.0.0:5000/ping

curl -XPOST {ADDRESS:PORT}/v1/predict -F 'file=@{FILE PATH}'
# ex) curl -XPOST 0.0.0.0:5000/v1/predict -F 'file=@kitten.jpg'

# 혹시 윈도우라면..
# ex) curl -XPOST 0.0.0.0:5000/v1/predict -F "file=@kitten.jpg"
```