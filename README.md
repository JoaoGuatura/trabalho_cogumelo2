# trabalho_cogumelo2
# Resultados Treinamentos

## Requisitos

- Python 3.11.6
- pip

## Como rodar

# Criar ambiente virtual
python -m venv venv
.\venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Rodar o pipeline
python main.py train
python main.py visualize --clean-data data/mushrooms_edited.csv

Model  Accuracy     AUC  Recall   Prec.  \ <br>
knn                K Neighbors Classifier    1.0000  1.0000  1.0000  1.0000 <br>
svm                   SVM - Linear Kernel    1.0000  1.0000  1.0000  1.0000 <br>
rf               Random Forest Classifier    1.0000  1.0000  1.0000  1.0000 <br>
qda       Quadratic Discriminant Analysis    1.0000  1.0000  1.0000  1.0000 <br>
et                 Extra Trees Classifier    1.0000  1.0000  1.0000  1.0000 <br>
lightgbm  Light Gradient Boosting Machine    1.0000  1.0000  1.0000  1.0000 <br>
lr                    Logistic Regression    0.9998  1.0000  0.9998  0.9998 <br>
dt               Decision Tree Classifier    0.9998  0.9998  0.9998  0.9998 <br>
ada                  Ada Boost Classifier    0.9998  1.0000  0.9998  0.9998 <br>
ridge                    Ridge Classifier    0.9996  1.0000  0.9996  0.9996 <br>
gbc          Gradient Boosting Classifier    0.9996  1.0000  0.9996  0.9996 <br>
lda          Linear Discriminant Analysis    0.9996  0.9993  0.9996  0.9996 <br>
nb                            Naive Bayes    0.9639  0.9968  0.9639  0.9666 <br>
dummy                    Dummy Classifier    0.5179  0.5000  0.5179  0.2683 <br>

              F1   Kappa     MCC  TT (Sec) <br>
knn       1.0000  1.0000  1.0000     0.535 <br>
svm       1.0000  1.0000  1.0000     0.162 <br>
rf        1.0000  1.0000  1.0000     0.224 <br>
qda       1.0000  1.0000  1.0000     0.193 <br>
et        1.0000  1.0000  1.0000     0.200 <br>
lightgbm  1.0000  1.0000  1.0000     0.287 <br>
lr        0.9998  0.9996  0.9996     0.800 <br>
dt        0.9998  0.9996  0.9996     0.171 <br>
ada       0.9998  0.9996  0.9996     0.311 <br>
ridge     0.9996  0.9993  0.9993     0.166 <br>
gbc       0.9996  0.9993  0.9993     0.294 <br>
lda       0.9996  0.9993  0.9993     0.191 <br>
nb        0.9639  0.9280  0.9305     0.176 <br>
dummy     0.3535  0.0000  0.0000     0.168 <br>
Best Classification Model: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', <br>
                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2, <br>
                     weights='uniform') <br>
Training complete. Model object: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', <br>
                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2, <br>
                     weights='uniform') <br>

------------------------------------------------------------

2025-04-13 19:58:53,982 [INFO] Training a model on mushrooms.csv | Target: class | Task: classification <br>
                    Description             Value <br>
0                    Session id               123 <br>
1                        Target             class <br>
2                   Target type            Binary <br>
3                Target mapping        e: 0, p: 1 <br>
4           Original data shape        (8124, 23) <br>
5        Transformed data shape       (8124, 113) <br>
6   Transformed train set shape       (5686, 113) <br>
7    Transformed test set shape       (2438, 113) <br>
8          Categorical features                22 <br>
9                    Preprocess              True <br>
10              Imputation type            simple <br>
11           Numeric imputation              mean <br>
12       Categorical imputation              mode <br>
13     Maximum one-hot encoding                25 <br>
14              Encoding method              None <br>
15               Fold Generator   StratifiedKFold <br>
16                  Fold Number                10 <br>
17                     CPU Jobs                -1 <br>
18                      Use GPU             False <br>
19               Log Experiment             False <br>
20              Experiment Name  clf-default-name <br>
21                          USI              f91d <br>
