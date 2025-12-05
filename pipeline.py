#!/usr/bin/env python3
"""
Pipeline de Reconhecimento Facial - Sistema de Controle de Frequência

Este script centraliza todas as operações do sistema:
- Processamento de imagens (detecção + normalização)
- Testes de acurácia
- Identificação em cenário real
"""

import argparse
import sys
from pathlib import Path

# Adiciona o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.processador import ProcessadorImagens
from src.testes import executar_testes_acuracia, gerar_relatorio_markdown
from src.identificacao import processar_cenario_real, processar_imagem_individual


def comando_processar(args):
    """Processa imagens do dataset aplicando normalização."""
    print("=" * 60)
    print("PROCESSAMENTO DE DATASET")
    print("=" * 60)
    
    processador = ProcessadorImagens(args.input, args.output)
    metodos = args.metodos.split(',') if args.metodos else ['clahe', 'histogram']
    
    stats = processador.processar_todas(
        metodos=metodos,
        skip_existing=not args.force
    )
    
    print(f"\n✓ Processamento concluído!")
    return stats


def comando_testar(args):
    """Executa testes de acurácia."""
    print("=" * 60)
    print("TESTES DE ACURÁCIA")
    print("=" * 60)
    
    res_clahe, res_hist = executar_testes_acuracia(
        data_dir=args.data_dir,
        db_clahe=args.db_clahe,
        db_histogram=args.db_histogram,
        threshold=args.threshold
    )
    
    if args.output:
        gerar_relatorio_markdown(res_clahe, res_hist, args.output)
        print(f"\n✓ Relatório salvo em: {args.output}")
    
    return res_clahe, res_hist


def comando_identificar(args):
    """Identifica rostos em imagens de cenário real."""
    print("=" * 60)
    print("IDENTIFICAÇÃO - CENÁRIO REAL")
    print("=" * 60)
    
    if args.imagem:
        # Processa uma única imagem
        resultado = processar_imagem_individual(
            img_path=args.imagem,
            db_path=args.database,
            output_path=args.output,
            threshold=args.threshold
        )
        
        if resultado:
            print(f"\n✓ Imagem anotada salva em: {args.output}")
            print(f"  - Faces detectadas: {resultado['total_faces']}")
            print(f"  - Identificadas: {resultado['identificados']}")
    
    elif args.batch:
        # Processa múltiplas imagens
        imagens = args.batch.split(',')
        resultados = processar_cenario_real(
            imagens_alvo=imagens,
            db_path=args.database,
            output_dir=args.output_dir,
            threshold=args.threshold
        )
        
        print(f"\n✓ Processadas {len(resultados)} imagens")
        print(f"  Resultados salvos em: {args.output_dir}")
    
    else:
        print("Erro: especifique --imagem ou --batch")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Reconhecimento Facial para Controle de Frequência",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Processar dataset com CLAHE e Histogram
  python pipeline.py processar --input data/images --output data/imagens_processadas

  # Processar apenas com CLAHE
  python pipeline.py processar --metodos clahe

  # Executar testes de acurácia
  python pipeline.py testar --output RELATORIO_TESTES.md

  # Identificar rostos em uma imagem
  python pipeline.py identificar --imagem foto_turma.jpg --output resultado.jpg

  # Processar múltiplas imagens de teste
  python pipeline.py identificar --batch "im1.jpg,im2.jpg,im3.jpg" --output-dir resultados/
        """
    )
    
    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponíveis')
    
    # Comando: processar
    parser_processar = subparsers.add_parser('processar', help='Processar dataset de imagens')
    parser_processar.add_argument('--input', default='data/images', help='Diretório de entrada')
    parser_processar.add_argument('--output', default='data/imagens_processadas', help='Diretório de saída')
    parser_processar.add_argument('--metodos', help='Métodos separados por vírgula (ex: clahe,histogram)')
    parser_processar.add_argument('--force', action='store_true', help='Reprocessar imagens já processadas')
    parser_processar.set_defaults(func=comando_processar)
    
    # Comando: testar
    parser_testar = subparsers.add_parser('testar', help='Executar testes de acurácia')
    parser_testar.add_argument('--data-dir', default='data/images', help='Diretório com imagens de teste')
    parser_testar.add_argument('--db-clahe', default='data/imagens_processadas/clahe', help='Base CLAHE')
    parser_testar.add_argument('--db-histogram', default='data/imagens_processadas/histogram', help='Base Histogram')
    parser_testar.add_argument('--threshold', type=float, default=0.6, help='Limiar de distância')
    parser_testar.add_argument('--output', help='Arquivo de saída (Markdown)')
    parser_testar.set_defaults(func=comando_testar)
    
    # Comando: identificar
    parser_identificar = subparsers.add_parser('identificar', help='Identificar rostos em imagens')
    parser_identificar.add_argument('--imagem', help='Imagem individual para processar')
    parser_identificar.add_argument('--batch', help='Múltiplas imagens separadas por vírgula')
    parser_identificar.add_argument('--database', default='data/imagens_processadas/clahe', help='Base de dados')
    parser_identificar.add_argument('--output', default='resultado_anotado.jpg', help='Arquivo de saída (imagem única)')
    parser_identificar.add_argument('--output-dir', default='data/resultados_cenario_real', help='Diretório de saída (batch)')
    parser_identificar.add_argument('--threshold', type=float, default=0.6, help='Limiar de distância')
    parser_identificar.set_defaults(func=comando_identificar)
    
    args = parser.parse_args()
    
    if not args.comando:
        parser.print_help()
        return
    
    # Executa o comando
    args.func(args)


if __name__ == "__main__":
    main()
