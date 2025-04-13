# trabalho_cogumelo2
# Resultados Treinamentos

Model  Accuracy     AUC  Recall   Prec.  \
knn                K Neighbors Classifier    1.0000  1.0000  1.0000  1.0000
svm                   SVM - Linear Kernel    1.0000  1.0000  1.0000  1.0000
rf               Random Forest Classifier    1.0000  1.0000  1.0000  1.0000
qda       Quadratic Discriminant Analysis    1.0000  1.0000  1.0000  1.0000
et                 Extra Trees Classifier    1.0000  1.0000  1.0000  1.0000
lightgbm  Light Gradient Boosting Machine    1.0000  1.0000  1.0000  1.0000
lr                    Logistic Regression    0.9998  1.0000  0.9998  0.9998
dt               Decision Tree Classifier    0.9998  0.9998  0.9998  0.9998
ada                  Ada Boost Classifier    0.9998  1.0000  0.9998  0.9998
ridge                    Ridge Classifier    0.9996  1.0000  0.9996  0.9996
gbc          Gradient Boosting Classifier    0.9996  1.0000  0.9996  0.9996
lda          Linear Discriminant Analysis    0.9996  0.9993  0.9996  0.9996
nb                            Naive Bayes    0.9639  0.9968  0.9639  0.9666
dummy                    Dummy Classifier    0.5179  0.5000  0.5179  0.2683

              F1   Kappa     MCC  TT (Sec)
knn       1.0000  1.0000  1.0000     0.535
svm       1.0000  1.0000  1.0000     0.162
rf        1.0000  1.0000  1.0000     0.224
qda       1.0000  1.0000  1.0000     0.193
et        1.0000  1.0000  1.0000     0.200
lightgbm  1.0000  1.0000  1.0000     0.287
lr        0.9998  0.9996  0.9996     0.800
dt        0.9998  0.9996  0.9996     0.171
ada       0.9998  0.9996  0.9996     0.311
ridge     0.9996  0.9993  0.9993     0.166
gbc       0.9996  0.9993  0.9993     0.294
lda       0.9996  0.9993  0.9993     0.191
nb        0.9639  0.9280  0.9305     0.176
dummy     0.3535  0.0000  0.0000     0.168
Best Classification Model: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                     weights='uniform')
Training complete. Model object: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                     weights='uniform')

------------------------------------------------------------

2025-04-13 19:58:53,982 [INFO] Training a model on mushrooms.csv | Target: class | Task: classification
                    Description             Value
0                    Session id               123
1                        Target             class
2                   Target type            Binary
3                Target mapping        e: 0, p: 1
4           Original data shape        (8124, 23)
5        Transformed data shape       (8124, 113)
6   Transformed train set shape       (5686, 113)
7    Transformed test set shape       (2438, 113)
8          Categorical features                22
9                    Preprocess              True
10              Imputation type            simple
11           Numeric imputation              mean
12       Categorical imputation              mode
13     Maximum one-hot encoding                25
14              Encoding method              None
15               Fold Generator   StratifiedKFold
16                  Fold Number                10
17                     CPU Jobs                -1
18                      Use GPU             False
19               Log Experiment             False
20              Experiment Name  clf-default-name
21                          USI              f91d
