import pandas as pd
from ports.training_port import TrainingPort
from pycaret.classification import (
    setup as class_setup,
    compare_models as class_compare,
    create_model,
    save_model,
    plot_model,
    interpret_model
)
import matplotlib.pyplot as plt
import os

class PyCaretAdapter(TrainingPort):
    def train_model(self, df: pd.DataFrame, target: str, task_type: str):
        """
        Treina o modelo e gera artefatos (plots, modelo salvo)
        """
        if task_type == "classification":
            # Configuração do experimento
            exp = class_setup(
                data=df,
                target=target,
                session_id=123,
                html=False,
                log_plots=True,
                verbose=False
            )
            
            # Comparação de modelos
            best_model = class_compare(sort='AUC', n_select=3)
            
            # Seleciona o SVM Linear (mais rápido entre os melhores)
            svm_model = create_model('svm')
            
            # Geração de plots
            self._generate_plots(svm_model)
            
            # Interpretação do modelo (SHAP)
            self._interpret_model(svm_model)
            
            # Salva o modelo
            os.makedirs('models', exist_ok=True)
            save_model(svm_model, 'models/best_mushroom_model')
            
            return svm_model
        
        else:
            raise ValueError("Esta implementação foca apenas em classificação")

    def _generate_plots(self, model):
        """Gera e salva visualizações importantes"""
        plots = {
            'auc': 'ROC Curve',
            'confusion_matrix': 'Confusion Matrix',
            'feature': 'Feature Importance'
        }
        
        for plot_type, name in plots.items():
            try:
                plot_model(model, plot=plot_type, save=True, scale=2)
                plt.title(f"{name} - SVM Linear")
                plt.savefig(f"reports/{plot_type}_plot.png", bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Erro ao gerar plot {plot_type}: {str(e)}")

    def _interpret_model(self, model):
        """Gera análise SHAP e salva"""
        try:
            interpret_model(model, save='reports/shap_summary.html')
        except Exception as e:
            print(f"Erro na interpretação: {str(e)}")