import pandas as pd
import argparse
import sys
import logging
import os
from pathlib import Path

# Adapters
from adapters.kaggle_downloader_adapter import KaggleDownloaderAdapter
from adapters.ydata_profiling_adapter import YDataProfilingAdapter
from adapters.dtale_adapter import DtaleAdapter
from adapters.pycaret_adapter import PyCaretAdapter
from visualizations import generate_visualizations

# Application
from application.use_cases import MLUseCases

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Cria estrutura de diretórios necessária"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def main():
    setup_directories()
    
    parser = argparse.ArgumentParser(description="Pipeline de ML para classificação de cogumelos")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Comando para pré-processamento
    preprocess_parser = subparsers.add_parser("preprocess", help="Pré-processa o dataset original")
    preprocess_parser.add_argument("input_csv", help="Caminho para o arquivo mushrooms.csv original")
    
    # Comando para treinamento
    train_parser = subparsers.add_parser("train", help="Treina o modelo SVM Linear")
    train_parser.add_argument("--clean-data", default="mushrooms_edited.csv",
                            help="Caminho para os dados pré-processados")

    # Comando para visualização
    viz_parser = subparsers.add_parser("visualize", help="Gera gráficos exploratórios")
    viz_parser.add_argument("--clean-data", default="mushrooms_edited.csv",
                            help="Caminho para os dados pré-processados")
    
    args = parser.parse_args()
    
    # Inicializa adapters
    dtale_adapter = DtaleAdapter()
    training_adapter = PyCaretAdapter()
    
    ml_use_cases = MLUseCases(
        dataset_adapter=None,
        profiler_adapter=None,
        dtale_adapter=dtale_adapter,
        training_adapter=training_adapter
    )
    
    if args.command == "preprocess":
        logger.info("Iniciando pré-processamento...")
        ml_use_cases.preprocess_data(args.input_csv)
        logger.info("Pré-processamento concluído. Dados salvos em data/mushrooms_clean.csv")
        
    elif args.command == "train":
        logger.info("Iniciando treinamento do SVM Linear...")
        ml_use_cases.train_model(args.clean_data, target_col='class', task_type='classification')

        logger.info("""
        Treinamento concluído!
        Modelo salvo em: models/best_mushroom_model
        Relatórios em: reports/
        """)

    elif args.command == "visualize":
        logger.info("Gerando visualizações exploratórias...")
        if not os.path.exists(args.clean_data):
            logger.error(f"Arquivo não encontrado: {args.clean_data}")
            sys.exit(1)
        df = pd.read_csv(args.clean_data)
        generate_visualizations(df)
        logger.info("Visualizações salvas na pasta 'reports/'.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Falha na execução: {str(e)}")
        sys.exit(1)
