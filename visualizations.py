import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import LabelEncoder

def generate_visualizations(df: pd.DataFrame):
    # Separar X e y
    X = df.drop('class', axis=1)
    y = df['class']

    # Codificar variáveis categóricas
    X_encoded = pd.get_dummies(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_encoded)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=y, palette='Set1')
    plt.title('PCA - Projeção 2D dos Cogumelos')
    plt.tight_layout()
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(X_encoded)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=y, palette='Set1')
    plt.title('t-SNE - Projeção 2D dos Cogumelos')
    plt.tight_layout()
    plt.show()

    # UMAP
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(X_encoded)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], hue=y, palette='Set1')
    plt.title('UMAP - Projeção 2D dos Cogumelos')
    plt.tight_layout()
    plt.show()

    # Heatmap de Correlação com a classe
    df_corr = X_encoded.copy()
    df_corr['class'] = y_encoded
    correlation = df_corr.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation[['class']].sort_values(by='class', ascending=False), annot=True, cmap='coolwarm')
    plt.title('Correlação das Variáveis com a Classe')
    plt.show()

    # Barplot de uma variável categórica (exemplo: odor)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='odor', hue='class', palette='Set2')
    plt.title('Distribuição de Odores por Classe')
    plt.tight_layout()
    plt.show()
