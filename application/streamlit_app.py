import streamlit as st
import pandas as pd
from pycaret.classification import setup as setup_clf, compare_models as compare_models_clf
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg

st.title("AutoML com PyCaret")

# 1. Upload do dataset
file = st.file_uploader("Faça upload do seu dataset (formato CSV)", type=["csv"])

if file:
    try:
        df = pd.read_csv(file)
        st.write("Prévia do Dataset:", df.head())

        # 2. Validação básica
        if df.empty:
            st.error("Erro: O dataset está vazio.")
        elif df.shape[1] < 2:
            st.error("Erro: O dataset deve ter pelo menos duas colunas.")
        else:
            # 3. Seleção da variável alvo
            target = st.selectbox("Escolha a variável alvo (target):", df.columns)

            if target:
                tipo_problema = "Classificação" if df[target].dtype == "object" or df[target].nunique() < 20 else "Regressão"
                st.write(f"Tipo de problema detectado: **{tipo_problema}**")

                if st.button("Iniciar Modelagem"):
                    with st.spinner("Executando PyCaret..."):
                        if tipo_problema == "Classificação":
                            setup_clf(data=df, target=target, silent=True, use_gpu=False)
                            best_model = compare_models_clf()
                        else:
                            setup_reg(data=df, target=target, silent=True, use_gpu=False)
                            best_model = compare_models_reg()

                    st.success("Modelagem finalizada!")
                    st.write("Melhor modelo encontrado:", best_model)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")