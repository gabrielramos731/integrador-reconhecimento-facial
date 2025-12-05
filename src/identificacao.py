"""
Módulo de identificação facial em cenário real.
"""

import os
import cv2
import time
from pathlib import Path
from src.preprocessamento import get_detector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Aviso: DeepFace não está instalado. Identificação não disponível.")


def extrair_id(nome_arquivo):
    """Extrai o ID da pessoa a partir do nome do arquivo."""
    base = os.path.basename(nome_arquivo)
    return os.path.splitext(base)[0].split('-')[0]


def garantir_diretorio(path):
    """Cria diretório se não existir."""
    if not os.path.exists(path):
        os.makedirs(path)


def processar_imagem_individual(img_path, db_path, output_path="resultado_anotado.jpg", threshold=0.6):
    """
    Processa uma única imagem, identifica rostos e gera imagem anotada.
    
    Args:
        img_path: Caminho da imagem de entrada
        db_path: Caminho da base de dados processada
        output_path: Caminho da imagem de saída
        threshold: Limiar de distância para aceitação
        
    Returns:
        dict: Estatísticas do processamento
    """
    if not DEEPFACE_AVAILABLE:
        print("Erro: DeepFace não está instalado.")
        return None
    
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
        cor = (0, 0, 255)  # Vermelho
        
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
                    df_filtered = df[df['distance'] < threshold]
                    if not df_filtered.empty:
                        best_match = df_filtered.iloc[0]
                        identity_file = os.path.basename(best_match['identity'])
                        nome_identificado = extrair_id(identity_file)
                        distancia = best_match['distance']
                        cor = (0, 255, 0)  # Verde
                        identificados += 1
        except Exception as e:
            print(f"Erro na identificação: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Anotar imagem com bordas e texto mais largos
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
    cv2.imwrite(output_path, img_anotada)
    
    print(f"✓ {total_faces} faces detectadas, {identificados} identificadas")
    
    return {
        'arquivo': os.path.basename(img_path),
        'total_faces': total_faces,
        'identificados': identificados,
        'detalhes': detalhes_identificacao,
        'path_saida': output_path
    }


def processar_cenario_real(imagens_alvo, db_path, output_dir="data/resultados_cenario_real", threshold=0.6):
    """
    Processa múltiplas imagens de cenário real.
    
    Args:
        imagens_alvo: Lista de caminhos de imagens
        db_path: Caminho da base de dados
        output_dir: Diretório de saída
        threshold: Limiar de distância
        
    Returns:
        list: Lista de resultados
    """
    garantir_diretorio(output_dir)
    
    resultados = []
    for img_path in imagens_alvo:
        if os.path.exists(img_path):
            nome_arquivo = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"anotada_{nome_arquivo}")
            
            res = processar_imagem_individual(img_path, db_path, output_path, threshold)
            if res:
                resultados.append(res)
        else:
            print(f"Imagem não encontrada: {img_path}")

    return resultados


def gerar_relatorio_cenario_real(resultados, output_file="RELATORIO_CENARIO_REAL.md"):
    """Gera relatório Markdown dos resultados de cenário real."""
    texto_relatorio = "# Relatório - Teste em Cenário Real (Sala de Aula)\n\n"
    texto_relatorio += f"**Data:** {time.strftime('%d/%m/%Y')}\n\n"
    texto_relatorio += "Nesta etapa, o sistema foi submetido a imagens de ambiente real (sala de aula), contendo múltiplos indivíduos, variações de pose, iluminação não controlada e oclusões parciais. O método de normalização utilizado foi o **CLAHE**, dado seu melhor desempenho nos testes controlados.\n\n"
    
    for idx, res in enumerate(resultados, 1):
        texto_relatorio += f"## {idx}. Análise da Imagem: {res['arquivo']}\n\n"
        texto_relatorio += f"- **Total de Faces Detectadas:** {res['total_faces']}\n"
        texto_relatorio += f"- **Indivíduos Identificados:** {res['identificados']}\n"
        taxa = (res['identificados']/res['total_faces']*100) if res['total_faces'] > 0 else 0
        texto_relatorio += f"- **Taxa de Reconhecimento:** {taxa:.1f}%\n\n"
        
        texto_relatorio += "| Face Detectada | Identidade Atribuída | Confiança (1-dist) | Status |\n"
        texto_relatorio += "|---|---|---|---|\n"
        
        for det in res['detalhes']:
            status = "✅ Identificado" if det['identificado'] != "Desconhecido" else "⚠️ Desconhecido"
            confianca = f"{1-det['distancia']:.2f}" if det['identificado'] != "Desconhecido" else "-"
            texto_relatorio += f"| {det['bbox']} | **{det['identificado']}** | {confianca} | {status} |\n"
        
        texto_relatorio += f"\n![Resultado {res['arquivo']}]({res['path_saida']})\n\n"

    with open(output_file, "w") as f:
        f.write(texto_relatorio)
    
    print(f"\nRelatório salvo em: {output_file}")
