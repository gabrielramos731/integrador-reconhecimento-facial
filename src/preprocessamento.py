"""
Módulo de preprocessamento de imagens para reconhecimento facial.
"""

import os
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from mtcnn.mtcnn import MTCNN

try:
    from pillow_heif import register_heif_opener
    from PIL import Image
    register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False


_detector = None


def get_detector():
    """Retorna uma instância compartilhada do detector MTCNN."""
    global _detector
    if _detector is None:
        _detector = MTCNN()
    return _detector


def alinhar_rosto_com_mtcnn(caminho_imagem):
    """Detecta e recorta o rosto principal da imagem."""
    try:
        img = cv2.imread(caminho_imagem)
        
        if img is None and HEIF_SUPPORT and caminho_imagem.lower().endswith('.heic'):
            try:
                pil_img = Image.open(caminho_imagem)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                return None
        
        if img is None:
            return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = get_detector()
        resultados = detector.detect_faces(img)
        
        if resultados:
            x1, y1, largura, altura = resultados[0]['box']
            x2, y2 = x1 + largura, y1 + altura
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            rosto_recortado = img[y1:y2, x1:x2]
            return cv2.cvtColor(rosto_recortado, cv2.COLOR_RGB2BGR)
        else:
            return None
    except Exception:
        return None


def normalizar_iluminacao(imagem_rosto, method="clahe"):
    """Aplica normalização de iluminação (CLAHE ou Histogram)."""
    if method == "histogram":
        img_gray = cv2.cvtColor(imagem_rosto, cv2.COLOR_BGR2GRAY)
        img_norm = cv2.equalizeHist(img_gray)
        return cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = cv2.cvtColor(imagem_rosto, cv2.COLOR_BGR2GRAY)
        img_norm = clahe.apply(img_gray)
        return cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    return imagem_rosto


def preprocessamento_base(caminho_imagem, metodo_normalizacao="clahe"):
    """Pipeline: detecta rosto e normaliza iluminação."""
    rosto_alinhado = alinhar_rosto_com_mtcnn(caminho_imagem)
    if rosto_alinhado is not None:
        return normalizar_iluminacao(rosto_alinhado, method=metodo_normalizacao)
    return None
