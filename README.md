# Sistema de Reconhecimento Facial - Controle de Frequência

Sistema automatizado de reconhecimento facial para controle de frequência em sala de aula. O projeto implementa detecção de rostos, normalização de imagem e identificação facial utilizando redes neurais profundas.

## 📋 Características

- **Detecção de Faces**: Utiliza MTCNN (Multi-task Cascaded Convolutional Networks)
- **Normalização de Iluminação**: Suporta CLAHE e Equalização de Histograma
- **Reconhecimento Facial**: Implementado com VGG-Face (DeepFace)
- **Testes de Acurácia**: Validação automática do sistema
- **Identificação em Cenário Real**: Processa fotos de turmas com múltiplos indivíduos

## 🗂️ Estrutura do Projeto

```
integrador/
├── pipeline.py                  # Script principal (CLI)
├── src/
│   ├── preprocessamento.py      # Detecção e normalização de faces
│   ├── processador.py           # Processamento em lote
│   ├── testes.py                # Testes de acurácia
│   └── identificacao.py         # Identificação em cenário real
├── data/
│   ├── images/                  # Imagens originais do dataset
│   ├── imagens_processadas/     # Imagens processadas
│   │   ├── clahe/               # Normalizadas com CLAHE
│   │   └── histogram/           # Normalizadas com Histogram
│   └── resultados_cenario_real/ # Resultados de identificação
├── requirements.txt             # Dependências do projeto
└── README.md                    # Este arquivo
```

## 🚀 Instalação

### 1. Clonar o Repositório

```bash
git clone https://github.com/gabrielramos731/integrador-reconhecimento-facial.git
cd integrador-reconhecimento-facial
```

### 2. Criar Ambiente Virtual

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### Dependências Principais

- **OpenCV**: Processamento de imagem
- **MTCNN**: Detecção de faces
- **TensorFlow**: Backend para redes neurais
- **DeepFace**: Framework de reconhecimento facial
- **Pillow + pillow-heif**: Suporte a formatos de imagem (HEIC)

## 📖 Guia de Uso

O sistema é controlado através do script `pipeline.py`, que oferece três comandos principais:

### 1. Processar Dataset

Processa imagens do dataset aplicando técnicas de normalização.

```bash
# Processar com CLAHE e Histogram (padrão)
python pipeline.py processar

# Processar apenas com CLAHE
python pipeline.py processar --metodos clahe

# Especificar diretórios customizados
python pipeline.py processar --input data/images --output data/processadas

# Forçar reprocessamento (ignorar cache)
python pipeline.py processar --force
```

**O que faz:**
- Detecta rostos nas imagens
- Recorta e alinha as faces
- Aplica normalização de iluminação (CLAHE e/ou Histogram)
- Salva em `data/imagens_processadas/`

### 2. Identificar em Cenário Real

Identifica rostos em fotos de turmas ou ambientes reais.

```bash
# Processar uma única imagem
python pipeline.py identificar --imagem foto_turma.jpg --output resultado.jpg

# Processar múltiplas imagens
python pipeline.py identificar \
  --batch "im1.jpg,im2.jpg,im3.jpg" \
  --output-dir resultados/

# Especificar base de dados e threshold
python pipeline.py identificar \
  --imagem turma.jpg \
  --database data/imagens_processadas/clahe \
  --threshold 0.6
```

**O que faz:**
- Detecta todos os rostos na imagem
- Identifica cada pessoa contra a base de dados
- Gera imagem anotada com:
  - Caixas delimitadoras (verde = identificado, vermelho = desconhecido)
  - Nome da pessoa + nível de confiança

## 🔬 Metodologia

### Pré-processamento

1. **Detecção**: MTCNN detecta e recorta faces
2. **Normalização**: Duas técnicas disponíveis:
   - **CLAHE**: Equalização adaptativa por regiões (melhor para iluminação irregular)
   - **Histogram**: Equalização global (melhor para contraste uniforme)

### Reconhecimento

- **Modelo**: VGG-Face (rede neural convolucional)
- **Método**: Comparação de embeddings faciais
- **Métrica**: Distância euclidiana entre vetores de características
- **Threshold padrão**: 0.6 (valores menores = maior certeza)

## 📊 Exemplos de Uso

### Fluxo Completo

```bash
# 1. Processar dataset
python pipeline.py processar --metodos clahe,histogram

# 2. Executar testes de acurácia
python pipeline.py testar --output RELATORIO.md

# 3. Identificar alunos em foto de turma
python pipeline.py identificar --imagem turma_2025.jpg --output presenca.jpg
```

### Apenas Reconhecimento (Base já Processada)

```bash
python pipeline.py identificar \
  --imagem aula_hoje.jpg \
  --database data/imagens_processadas/clahe \
  --output frequencia_hoje.jpg
```
## 📝 Formatos Suportados

- **Imagens**: JPEG, PNG, HEIC


## 📚 Referências

- [DeepFace](https://github.com/serengil/deepface)
- [MTCNN](https://github.com/ipazc/mtcnn)
- [VGG-Face](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
- [OpenCV CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
