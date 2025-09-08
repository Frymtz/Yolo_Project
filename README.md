```markdown
# YOLO - Detecção de Componentes Eletrônicos

Este projeto implementa uma **YOLOv8** para detectar e contar **resistores, capacitores e transistores** em imagens, vídeos ou em tempo real via câmera.

---

## Estrutura de Pastas

```
YOLO-Components/
│── datasets/                # Datasets no formato YOLO
│   └── components/          # Ex: resistor, capacitor, transistor
│
│── scripts/                 # Scripts do projeto
│   ├── train.py             # Script de treinamento
│   ├── infer\_image.py       # Inferência em imagem
│   ├── infer\_video.py       # Inferência em vídeo
│   ├── infer\_count.py       # Inferência com real 
│
│── runs/                    # Resultados de treinamentos
│
│── main.py                  # Script principal com argparse
│── requirements.txt         # Dependências do projeto
│── README.md                # Este arquivo

````

---

## Configuração do Ambiente

1. Clone este repositório:

```bash
git clone https://github.com/Frymtz/Yolo_Project.git
cd yolo-components
````

2. Crie o ambiente virtual e instale dependências:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### Dependências principais

* `ultralytics` (YOLOv8)
* `opencv-python`
* `numpy`

---

## Dataset

Você pode usar datasets públicos de componentes eletrônicos (Roboflow, Kaggle, MDPI, etc.) ou montar o seu próprio.

Estrutura esperada:

```
datasets/components/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Arquivo `components.yaml`:

```yaml
path: datasets/components
train: images/train
val: images/val
test: images/test

names:
  0: resistor
  1: capacitor
  2: transistor
```

---

## Como Usar

### 1. Treinamento

```bash
python main.py --mode train
```

### 2. Inferência em Imagem

```bash
python main.py --mode image --source data/teste.jpg --weights runs/detect/train/weights/best.pt
```

### 3. Inferência em Vídeo

```bash
python main.py --mode video --source data/video.mp4 --weights runs/detect/train/weights/best.pt
```


### 5. Inferência em Tempo Real (Câmera)

```bash
python main.py --mode realtime --source 0 --weights runs/detect/train/weights/best.pt
```

> `--source 0` = câmera padrão.
> Pode ser substituído pelo caminho de um vídeo ou endereço de câmera IP.

---

##  Observações

* Pressione **Q** para sair do modo vídeo/câmera.
* A contagem é exibida tanto na tela quanto no terminal (dependendo do script).
---
