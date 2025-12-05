import os
import cv2
import pandas as pd
from deepface import DeepFace
from src.preprocessamento import get_detector
import time
from pathlib import Path

# Configurações
DATA_DIR = "data/images"
DB_CLAHE = "data/imagens_processadas/clahe"
DB_HISTOGRAM = "data/imagens_processadas/histogram"
EXTENSOES_VALIDAS = {'.jpg', '.jpeg', '.png', '.heic', '.HEIC'}

def extrair_id(nome_arquivo):
    """Extrai o ID da pessoa a partir do nome do arquivo (ex: Habo1-1.jpg -> Habo1)."""
    base = os.path.basename(nome_arquivo)
    nome_sem_ext = os.path.splitext(base)[0]
    # Assume que o ID é a parte antes do primeiro hífen
    return nome_sem_ext.split('-')[0]

def buscar_rosto_silencioso(face_img, db_path, threshold=0.6):
    """Busca um rosto no banco de dados sem imprimir no console."""
    temp_path = f"temp_face_{time.time()}.jpg"
    cv2.imwrite(temp_path, face_img)
    
    try:
        # DeepFace.find pode gerar output, tentamos silenciar
        result = DeepFace.find(
            img_path=temp_path,
            db_path=db_path,
            enforce_detection=False,
            silent=True,
            model_name="VGG-Face",
            detector_backend="opencv" # Usando opencv para ser consistente com o projeto
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

def executar_testes():
    print("Iniciando bateria de testes...")
    
    imagens_teste = []
    for f in os.listdir(DATA_DIR):
        if os.path.splitext(f)[1] in EXTENSOES_VALIDAS:
            imagens_teste.append(os.path.join(DATA_DIR, f))
    
    imagens_teste.sort()
    
    resultados_clahe = []
    resultados_histogram = []
    
    detector = get_detector()
    
    total = len(imagens_teste)
    print(f"Total de imagens para teste: {total}")
    
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
        matches_clahe = buscar_rosto_silencioso(melhor_rosto, DB_CLAHE)
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
        matches_hist = buscar_rosto_silencioso(melhor_rosto, DB_HISTOGRAM)
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

    return resultados_clahe, resultados_histogram

def gerar_relatorio(res_clahe, res_hist):
    total = len(res_clahe)
    acertos_clahe = sum(1 for r in res_clahe if r['acerto'])
    acertos_hist = sum(1 for r in res_hist if r['acerto'])
    
    acc_clahe = (acertos_clahe / total * 100) if total > 0 else 0
    acc_hist = (acertos_hist / total * 100) if total > 0 else 0
    
    relatorio = f"""# Relatório de Testes de Reconhecimento Facial

**Data:** {time.strftime("%d/%m/%Y")}
**Total de Imagens Testadas:** {total}

## Resumo dos Resultados

| Método | Acertos | Erros | Acurácia |
|--------|---------|-------|----------|
| CLAHE | {acertos_clahe} | {total - acertos_clahe} | {acc_clahe:.2f}% |
| Histogram | {acertos_hist} | {total - acertos_hist} | {acc_hist:.2f}% |

## Detalhes - CLAHE

| Arquivo | ID Real | ID Identificado | Distância | Resultado |
|---------|---------|-----------------|-----------|-----------|
"""
    
    for r in res_clahe:
        status = "✅" if r['acerto'] else "❌"
        dist = f"{r['distancia']:.4f}" if r['distancia'] is not None else "-"
        relatorio += f"| {r['arquivo']} | {r['id_real']} | {r['identificado']} | {dist} | {status} |\n"
        
    relatorio += "\n## Detalhes - Histogram\n\n"
    relatorio += "| Arquivo | ID Real | ID Identificado | Distância | Resultado |\n"
    relatorio += "|---------|---------|-----------------|-----------|-----------|\n"
    
    for r in res_hist:
        status = "✅" if r['acerto'] else "❌"
        dist = f"{r['distancia']:.4f}" if r['distancia'] is not None else "-"
        relatorio += f"| {r['arquivo']} | {r['id_real']} | {r['identificado']} | {dist} | {status} |\n"

    with open("RELATORIO_TESTES.md", "w") as f:
        f.write(relatorio)
    
    print("\nRelatório gerado em RELATORIO_TESTES.md")

if __name__ == "__main__":
    res_clahe, res_hist = executar_testes()
    gerar_relatorio(res_clahe, res_hist)
