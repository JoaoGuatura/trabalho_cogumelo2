# application/use_cases.py
import os
import pandas as pd

from ports.dataset_port import DatasetPort
from ports.profiling_port import ProfilingPort
from ports.dtale_port import DtalePort
from ports.training_port import TrainingPort

DATA_FOLDER = "data"

from typing import Optional
import pandas as pd

class MLUseCases:
    def __init__(self, dataset_adapter, profiler_adapter, dtale_adapter, training_adapter):
        self.dtale_adapter = dtale_adapter
        self.training_adapter = training_adapter

    def preprocess_data(self, input_path: str):
        """Remove colunas irrelevantes e salva versão limpa"""
        try:
            df = pd.read_csv(input_path)
            
            # Remoção de colunas (ex: 'veil-type' tem apenas 1 valor único)
            df.drop(columns=['veil-type'], inplace=True, errors='ignore')
            
            # Salva o dataset pré-processado
            output_path = "mushrooms_edited.csv"
            df.to_csv(output_path, index=False)
            
            return output_path
        except Exception as e:
            raise ValueError(f"Falha no pré-processamento: {str(e)}")

    def download_dataset(self, kaggle_name: str, output_path: str):
        # pass both arguments to the adapter
        self.dataset_adapter.download_dataset(kaggle_name, output_path)
        print(f"Dataset '{kaggle_name}' downloaded to '{output_path}'.")

    
    def profile_data(self, csv_filename: str):
        full_path = os.path.join(DATA_FOLDER, csv_filename)
        df = pd.read_csv(full_path)
        self.profiler_adapter.generate_report(df)
    
    def edit_data(self, csv_filename: str) -> str:
        """Launch dtale, then (optionally) store the edited dataset."""
        full_path = os.path.join(DATA_FOLDER, csv_filename)
        df = pd.read_csv(full_path)
        
        new_df = self.dtale_adapter.open_in_dtale(df)
        # For demonstration, we do not know how to get user edits.
        # We'll just pretend it's edited in place and re-saved.
        edited_path = os.path.join(DATA_FOLDER, "edited_data.csv")
        new_df.to_csv(edited_path, index=False)
        print(f"Saved (edited) data to {edited_path}")
        return edited_path
    
    def train_model(self, csv_filename: str, target_col: str, task_type: str):
        full_path = csv_filename if os.path.isfile(csv_filename) else os.path.join(DATA_FOLDER, csv_filename)
        df = pd.read_csv(full_path)
        model = self.training_adapter.train_model(df, target_col, task_type)
        # we simply print or return the model
        print(f"Training complete. Model object: {model}")
