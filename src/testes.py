"""
Módulo de testes de acurácia do sistema de reconhecimento facial.
"""

import os
import cv2
import time
from pathlib import Path
from src.preprocessamento import get_detector

# Suprime warnings do DeepFace
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Aviso: DeepFace não está instalado. Testes de acurácia não disponíveis.")


EXTENSOES_VALIDAS = {'.jpg', '.jpeg', '.png', '.heic', '.HEIC'}


def extrair_id(nome_arquivo):
    """Extrai o ID da pessoa a partir do nome do arquivo (ex: Habo1-1.jpg -> Habo1)."""
    base = os.path.basename(nome_arquivo)
    nome_sem_ext = os.path.splitext(base)[0]
    return nome_sem_ext.split('-')[0]


def buscar_rosto_silencioso(face_img, db_path, threshold=0.6):
    """Busca um rosto no banco de dados sem imprimir no console."""
    if not DEEPFACE_AVAILABLE:
        return []
    
    temp_path = f"temp_face_{time.time()}.jpg"
    cv2.imwrite(temp_path, face_img)
    
    try:
        result = DeepFace.find(
            img_path=temp_path,
            db_path=db_path,
            enforce_detection=False,
            silent=True,
            model_name="VGG-Face",
            detector_backend="opencv"
        )
        
        matches = []
        if isinstance(result, list) and len(result) > 0:
            df = result[0]
            if not df.empty:
                df_filtered = df[df['distance'] < threshold]
                for _, row in df_filtered.iterrows():
                    identity_file = os.path.basename(row['identity'])
                    identity_id = extrair_id(identity_file)
                    matches.append({
                        'file': identity_file,
                        'id': identity_id,
                        'distance': row['distance']
                    })
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return matches
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return []


def executar_testes_acuracia(data_dir='data/images', db_clahe='data/imagens_processadas/clahe', 
                              db_histogram='data/imagens_processadas/histogram', threshold=0.6):
    """
    Executa bateria de testes de acurácia.
    
    Retorna:
        tuple: (resultados_clahe, resultados_histogram)
    """
    print("Iniciando bateria de testes...")
    
    imagens_teste = []
    for f in os.listdir(data_dir):
        if os.path.splitext(f)[1] in EXTENSOES_VALIDAS:
            imagens_teste.append(os.path.join(data_dir, f))
    
    imagens_teste.sort()
    
    resultados_clahe = []
    resultados_histogram = []
    
    detector = get_detector()
    
    total = len(imagens_teste)
    print(f"Total de imagens para teste: {total}\n")
    
    for i, img_path in enumerate(imagens_teste):
        print(f"Processando {i+1}/{total}: {os.path.basename(img_path)}")
        
        # Carregar imagem
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Erro ao ler {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detectar rosto
        deteccoes = detector.detect_faces(img_rgb)
        
        if not deteccoes:
            print("  Nenhum rosto detectado.")
            continue
            
        # Pega o maior rosto (assume que é o alvo)
        maior_area = 0
        melhor_rosto = None
        
        for det in deteccoes:
            x, y, w, h = det['box']
            area = w * h
            if area > maior_area:
                maior_area = area
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
                melhor_rosto = img[y1:y2, x1:x2]
        
        if melhor_rosto is None:
            continue
            
        id_real = extrair_id(img_path)
        
        # Teste CLAHE
        matches_clahe = buscar_rosto_silencioso(melhor_rosto, db_clahe, threshold)
        top_match_clahe = matches_clahe[0] if matches_clahe else None
        acerto_clahe = False
        if top_match_clahe:
            if top_match_clahe['id'] == id_real:
                acerto_clahe = True
        
        resultados_clahe.append({
            'arquivo': os.path.basename(img_path),
            'id_real': id_real,
            'identificado': top_match_clahe['id'] if top_match_clahe else "Nenhum",
            'distancia': top_match_clahe['distance'] if top_match_clahe else None,
            'acerto': acerto_clahe
        })
        
        # Teste Histogram
        matches_hist = buscar_rosto_silencioso(melhor_rosto, db_histogram, threshold)
        top_match_hist = matches_hist[0] if matches_hist else None
        acerto_hist = False
        if top_match_hist:
            if top_match_hist['id'] == id_real:
                acerto_hist = True
                
        resultados_histogram.append({
            'arquivo': os.path.basename(img_path),
            'id_real': id_real,
            'identificado': top_match_hist['id'] if top_match_hist else "Nenhum",
            'distancia': top_match_hist['distance'] if top_match_hist else None,
            'acerto': acerto_hist
        })

    print("\n✓ Testes concluídos")
    return resultados_clahe, resultados_histogram


def gerar_relatorio_markdown(res_clahe, res_hist, output_file="RELATORIO_TESTES.md"):
    """Gera relatório em formato Markdown."""
    total = len(res_clahe)
    acertos_clahe = sum(1 for r in res_clahe if r['acerto'])
    acertos_hist = sum(1 for r in res_hist if r['acerto'])
    
    acc_clahe = (acertos_clahe / total * 100) if total > 0 else 0
    acc_hist = (acertos_hist / total * 100) if total > 0 else 0
    
    relatorio = f"""# Relatório de Testes de Reconhecimento Facial

**Data:** {time.strftime("%d/%m/%Y")}
**Total de Imagens Testadas:** {total}

## 1. Metodologia

O presente experimento visa avaliar a robustez de algoritmos de reconhecimento facial frente a diferentes técnicas de normalização de imagem. A metodologia adotada divide-se em pré-processamento, extração de características e verificação de identidade.

### 1.1. Pré-processamento e Normalização
Foram geradas duas bases de conhecimento distintas a partir das imagens originais, aplicando-se técnicas de aprimoramento de contraste para mitigar variações de iluminação:
*   **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Técnica que divide a imagem em pequenas regiões (*tiles*) e aplica a equalização de histograma em cada uma, limitando o contraste para evitar a amplificação de ruído. Este método é particularmente eficaz para melhorar detalhes locais e corrigir iluminação desigual.
*   **Equalização de Histograma (Global):** Técnica que ajusta as intensidades dos pixels de forma global, distribuindo-as uniformemente por todo o histograma da imagem. Este método melhora o contraste global, mas pode suprimir detalhes em regiões muito claras ou muito escuras.

### 1.2. Arquitetura de Reconhecimento
O sistema utiliza a biblioteca **DeepFace** com o modelo **VGG-Face** para a extração de *embeddings* faciais (vetores de características).
*   **Detecção:** As faces são detectadas e alinhadas antes do processamento para garantir consistência geométrica.
*   **Comparação:** A identificação é realizada através do cálculo da distância vetorial entre a face de teste e as faces armazenadas na base de conhecimento.
*   **Critério de Aceitação:** Foi estabelecido um limiar de distância (*threshold*) de **0.6**. Distâncias inferiores a este valor indicam uma correspondência positiva (mesma identidade).

### 1.3. Procedimento de Teste
O protocolo experimental consistiu na submissão de **{total} imagens de teste** ao sistema. Para cada imagem, o fluxo de validação foi:
1.  Extração da face presente na imagem de teste.
2.  Busca da identidade correspondente na base processada com **CLAHE**.
3.  Busca da identidade correspondente na base processada com **Histograma**.
4.  Comparação da identidade predita com a identidade real (*Ground Truth*) obtida a partir da nomenclatura do arquivo.

## 2. Resultados Experimentais

### 2.1. Resumo Quantitativo

| Método | Acertos | Erros | Acurácia |
|--------|---------|-------|----------|
| CLAHE | {acertos_clahe} | {total - acertos_clahe} | {acc_clahe:.2f}% |
| Histogram | {acertos_hist} | {total - acertos_hist} | {acc_hist:.2f}% |

## 2.2. Detalhamento Experimental - CLAHE

| Arquivo | ID Real | ID Identificado | Distância | Resultado |
|---------|---------|-----------------|-----------|-----------|
"""
    
    for r in res_clahe:
        status = "✅" if r['acerto'] else "❌"
        dist = f"{r['distancia']:.4f}" if r['distancia'] is not None else "-"
        relatorio += f"| {r['arquivo']} | {r['id_real']} | {r['identificado']} | {dist} | {status} |\n"
        
    relatorio += "\n## 2.3. Detalhamento Experimental - Histogram\n\n"
    relatorio += "| Arquivo | ID Real | ID Identificado | Distância | Resultado |\n"
    relatorio += "|---------|---------|-----------------|-----------|-----------|\n"
    
    for r in res_hist:
        status = "✅" if r['acerto'] else "❌"
        dist = f"{r['distancia']:.4f}" if r['distancia'] is not None else "-"
        relatorio += f"| {r['arquivo']} | {r['id_real']} | {r['identificado']} | {dist} | {status} |\n"

    with open(output_file, "w") as f:
        f.write(relatorio)
    
    print(f"\nRelatório gerado em {output_file}")
