import os
import cv2
from deepface import DeepFace
from src.preprocessamento import get_detector
import time
from pathlib import Path

# Configurações
IMAGENS_ALVO = ["im1.jpg", "im2.jpg", "im3.jpg", "img_teste.jpeg"]
DB_PATH = "data/imagens_processadas/clahe"
OUTPUT_DIR = "data/resultados_cenario_real"
THRESHOLD = 0.6

def garantir_diretorio(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extrair_id(nome_arquivo):
    base = os.path.basename(nome_arquivo)
    return os.path.splitext(base)[0].split('-')[0]

def processar_imagem_cenario_real(img_path, db_path, output_dir):
    print(f"Processando {img_path}...")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao ler {img_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = get_detector()
    
    try:
        resultados_deteccao = detector.detect_faces(img_rgb)
    except Exception as e:
        print(f"Erro na detecção: {e}")
        return None

    rostos_detectados = []
    img_anotada = img.copy()
    
    total_faces = len(resultados_deteccao)
    identificados = 0
    
    detalhes_identificacao = []

    for i, resultado in enumerate(resultados_deteccao):
        x, y, w, h = resultado['box']
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        
        face_img = img[y1:y2, x1:x2]
        
        # Busca no DB
        temp_path = f"temp_face_{time.time()}_{i}.jpg"
        cv2.imwrite(temp_path, face_img)
        
        nome_identificado = "Desconhecido"
        distancia = 0.0
        cor = (0, 0, 255) # Vermelho
        
        try:
            result = DeepFace.find(
                img_path=temp_path,
                db_path=db_path,
                enforce_detection=False,
                silent=True,
                model_name="VGG-Face",
                detector_backend="opencv"
            )
            
            if isinstance(result, list) and len(result) > 0:
                df = result[0]
                if not df.empty:
                    df_filtered = df[df['distance'] < THRESHOLD]
                    if not df_filtered.empty:
                        best_match = df_filtered.iloc[0]
                        identity_file = os.path.basename(best_match['identity'])
                        nome_identificado = extrair_id(identity_file)
                        distancia = best_match['distance']
                        cor = (0, 255, 0) # Verde
                        identificados += 1
        except Exception as e:
            print(f"Erro na identificação: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Anotar imagem
        # Aumentando espessura da borda e tamanho do texto
        cv2.rectangle(img_anotada, (x1, y1), (x2, y2), cor, 8)
        label = f"{nome_identificado}"
        if nome_identificado != "Desconhecido":
            label += f" ({1-distancia:.2f})"
            
        cv2.putText(img_anotada, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, cor, 5)
        
        detalhes_identificacao.append({
            'bbox': (x1, y1, x2, y2),
            'identificado': nome_identificado,
            'distancia': distancia
        })

    # Salvar imagem anotada
    nome_arquivo = os.path.basename(img_path)
    path_saida = os.path.join(output_dir, f"anotada_{nome_arquivo}")
    cv2.imwrite(path_saida, img_anotada)
    
    return {
        'arquivo': nome_arquivo,
        'total_faces': total_faces,
        'identificados': identificados,
        'detalhes': detalhes_identificacao,
        'path_saida': path_saida
    }

def gerar_relatorio_cenario_real():
    garantir_diretorio(OUTPUT_DIR)
    
    resultados = []
    for img_name in IMAGENS_ALVO:
        if os.path.exists(img_name):
            res = processar_imagem_cenario_real(img_name, DB_PATH, OUTPUT_DIR)
            if res:
                resultados.append(res)
        else:
            print(f"Imagem não encontrada: {img_name}")

    # Gerar texto para o relatório
    texto_relatorio = "\n## 3. Teste em Cenário Real (Sala de Aula)\n\n"
    texto_relatorio += "Nesta etapa, o sistema foi submetido a imagens de ambiente real (sala de aula), contendo múltiplos indivíduos, variações de pose, iluminação não controlada e oclusões parciais. O método de normalização utilizado foi o **CLAHE**, dado seu melhor desempenho nos testes controlados.\n\n"
    
    for res in resultados:
        texto_relatorio += f"### 3.1. Análise da Imagem: {res['arquivo']}\n\n"
        texto_relatorio += f"- **Total de Faces Detectadas:** {res['total_faces']}\n"
        texto_relatorio += f"- **Indivíduos Identificados:** {res['identificados']}\n"
        texto_relatorio += f"- **Taxa de Reconhecimento:** {(res['identificados']/res['total_faces']*100) if res['total_faces'] > 0 else 0:.1f}%\n\n"
        
        texto_relatorio += "| Face Detectada | Identidade Atribuída | Confiança (1-dist) | Status |\n"
        texto_relatorio += "|---|---|---|---|\n"
        
        for det in res['detalhes']:
            status = "✅ Identificado" if det['identificado'] != "Desconhecido" else "⚠️ Desconhecido"
            confianca = f"{1-det['distancia']:.2f}" if det['identificado'] != "Desconhecido" else "-"
            texto_relatorio += f"| {det['bbox']} | **{det['identificado']}** | {confianca} | {status} |\n"
        
        texto_relatorio += "\n"
        # Nota: Em um relatório real, incluiríamos a imagem aqui.
        # texto_relatorio += f"![Resultado {res['arquivo']}]({res['path_saida']})\n\n"

    with open("RELATORIO_TESTES.md", "a") as f:
        f.write(texto_relatorio)
    
    print("Relatório atualizado com sucesso.")

if __name__ == "__main__":
    gerar_relatorio_cenario_real()
