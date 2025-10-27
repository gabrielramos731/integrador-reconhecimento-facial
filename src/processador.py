"""
Módulo para processamento em lote de imagens.
"""

import os
import cv2
from pathlib import Path
from src.preprocessamento import preprocessamento_base

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ProcessadorImagens:
    """Processa imagens em lote aplicando métodos de normalização."""
    
    def __init__(self, dir_entrada, dir_saida):
        self.dir_entrada = Path(dir_entrada)
        self.dir_saida = Path(dir_saida)
        self.extensoes_validas = {'.jpg', '.jpeg', '.png', '.heic', '.HEIC'}
    
    def listar_imagens(self):
        """Lista todas as imagens válidas no diretório."""
        imagens = []
        for arquivo in self.dir_entrada.iterdir():
            if arquivo.is_file() and arquivo.suffix in self.extensoes_validas:
                imagens.append(arquivo)
        return sorted(imagens)
    
    def processar_imagem(self, caminho_imagem, metodo):
        """Processa uma única imagem."""
        return preprocessamento_base(str(caminho_imagem), metodo_normalizacao=metodo)
    
    def salvar_imagem(self, imagem, nome_original, metodo):
        """Salva a imagem processada."""
        dir_metodo = self.dir_saida / metodo
        dir_metodo.mkdir(parents=True, exist_ok=True)
        nome_base = Path(nome_original).stem
        nome_saida = f"{nome_base}.jpg"
        caminho_saida = dir_metodo / nome_saida
        cv2.imwrite(str(caminho_saida), imagem)
        return caminho_saida
    
    def ja_processada(self, nome_arquivo, metodo):
        """Verifica se uma imagem já foi processada."""
        nome_base = Path(nome_arquivo).stem
        nome_saida = f"{nome_base}.jpg"
        caminho_saida = self.dir_saida / metodo / nome_saida
        return caminho_saida.exists()
    
    def processar_todas(self, metodos=['clahe', 'histogram'], skip_existing=True):
        """
        Processa todas as imagens do diretório de entrada com os métodos especificados.
        
        Argumentos:
        metodos (list): Lista de métodos a serem aplicados.
        skip_existing (bool): Se True, pula imagens já processadas.
        
        Retorna:
        Dicionário com estatísticas do processamento.
        """
        imagens = self.listar_imagens()
        total_imagens = len(imagens)
        
        print(f"Processando {total_imagens} imagens com métodos: {', '.join(metodos)}")
        
        estatisticas = {
            'total': total_imagens,
            'processadas': 0,
            'puladas': 0,
            'falhas': 0,
        }
        
        # Cache para armazenar o rosto detectado e evitar redetecção
        from src.preprocessamento import alinhar_rosto_com_mtcnn, normalizar_iluminacao
        
        for idx, caminho_imagem in enumerate(imagens, 1):
            print(f"[{idx}/{total_imagens}] {caminho_imagem.name}", end=" ")
            
            # Verifica se já foi processada em todos os métodos
            if skip_existing:
                todos_processados = all(self.ja_processada(caminho_imagem.name, metodo) for metodo in metodos)
                if todos_processados:
                    print("✓ (já processada)")
                    estatisticas['puladas'] += 1
                    continue
            
            # Detecta o rosto uma vez apenas
            rosto_alinhado = alinhar_rosto_com_mtcnn(str(caminho_imagem))
            
            if rosto_alinhado is None:
                print("✗ (sem rosto)")
                estatisticas['falhas'] += 1
                continue
            
            # Aplica cada método de normalização no rosto já detectado
            sucesso = True
            for metodo in metodos:
                if skip_existing and self.ja_processada(caminho_imagem.name, metodo):
                    continue
                
                try:
                    imagem_processada = normalizar_iluminacao(rosto_alinhado, method=metodo)
                    self.salvar_imagem(imagem_processada, caminho_imagem.name, metodo)
                except Exception:
                    sucesso = False
            
            if sucesso:
                print("✓")
                estatisticas['processadas'] += 1
        
        print(f"\nConcluído: {estatisticas['processadas']} processadas, {estatisticas['puladas']} puladas, {estatisticas['falhas']} falhas")
        return estatisticas


def processar_dataset(dir_entrada='data/images', dir_saida='data/imagens_processadas', skip_existing=True):
    """Processa todo o dataset."""
    processador = ProcessadorImagens(dir_entrada, dir_saida)
    return processador.processar_todas(skip_existing=skip_existing)
