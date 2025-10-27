"""
Sistema de Processamento de Imagens para Reconhecimento Facial
"""

from src.processador import processar_dataset


if __name__ == "__main__":
    print("Iniciando processamento...\n")
    processar_dataset()
    print("\nProcessamento concluído!")
