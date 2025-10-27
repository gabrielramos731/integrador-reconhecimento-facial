# Sistema de Reconhecimento Facial

Processamento de imagens para reconhecimento facial com detecção de rostos e normalização de iluminação.

## Estrutura

```
integrador/
├── main.py                      # Script principal
├── src/
│   ├── preprocessamento.py      # Detecção e normalização
│   └── processador.py           # Processamento em lote
└── data/
    ├── images/                  # Imagens originais
    └── imagens_processadas/     # Imagens processadas
        ├── clahe/               # Método CLAHE
        └── histogram/           # Método Histogram Equalization
```

## Uso

```bash
source .venv/bin/activate
python main.py
```

## Métodos

- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Histogram**: Equalização de histograma padrão

## Formatos Suportados

JPEG, PNG, HEIC

## Dependências

```bash
pip install opencv-python mtcnn tensorflow pillow-heif pillow
```
