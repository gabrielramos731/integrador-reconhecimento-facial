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
# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Executar processamento
python main.py
```

## Métodos

- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Histogram**: Equalização de histograma padrão

## Formatos Suportados

JPEG, PNG, HEIC
