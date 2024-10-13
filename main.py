import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Semente para o random_state
SEED = 42

# Carregando o dataset Breast Cancer
data = load_breast_cancer()
X = data.data
y = data.target

# Verificando se existem valores NaN ou infinitos nos dados
print("Existem valores NaN no dataset?:", np.isnan(X).any())
print("Existem valores infinitos no dataset?:", np.isinf(X).any())

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Definindo o pipeline que inclui padronização e o modelo
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),  # Padronização dos dados
        ("classifier", KNeighborsClassifier()),  # O modelo será definido dinamicamente
    ]
)

# Definindo os hiperparâmetros para cada modelo
param_grid = [
    # KNN
    {
        "classifier": [KNeighborsClassifier()],
        "classifier__n_neighbors": [3, 5, 7, 9, 11],
        "classifier__weights": ["uniform", "distance"],  # Adicionando diferentes ponderações
        "classifier__p": [1, 2],  # Adicionando a métrica de distância: 1 para Manhattan, 2 para Euclidiana
    },
    # SVM
    {
        "classifier": [SVC(random_state=SEED)],
        "classifier__C": [0.1, 1, 10, 100, 1000],
        "classifier__kernel": ["linear", "rbf", "poly"],  # Adicionando o kernel polinomial
        "classifier__gamma": ["scale", "auto"],  # Adicionando diferentes opções de gamma
        "classifier__degree": [2, 3],  # Se o kernel for polinomial, explorar diferentes graus
    },
    # Decision Tree
    {
        "classifier": [DecisionTreeClassifier(random_state=SEED)],
        "classifier__max_depth": [3, 5, 7, 10, None],  # Adicionando a opção de sem limite de profundidade
        "classifier__min_samples_split": [2, 5, 10],  # Adicionando restrições de divisão mínima
        "classifier__min_samples_leaf": [1, 2, 4],  # Adicionando a restrição mínima de amostras em folhas
    },
    # Random Forest
    {
        "classifier": [RandomForestClassifier(random_state=SEED)],
        "classifier__n_estimators": [50, 100, 200],  # Número de árvores
        "classifier__max_depth": [3, 5, 7, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__bootstrap": [True, False],  # Adicionando opção de uso de bootstrap
    },
    # Logistic Regression
    {
        "classifier": [LogisticRegression(random_state=SEED)],
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2", "elasticnet"],  # Diferentes penalidades
        "classifier__solver": ["liblinear", "saga"],  # Diferentes otimizadores
    },
]

# Realizando a busca por hiperparâmetros com GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Mostrando os melhores parâmetros e modelo
print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor modelo: {grid_search.best_estimator_}")

# Realizando predições no conjunto de teste com o melhor modelo
y_pred = grid_search.best_estimator_.predict(X_test)

# Exibindo o relatório de classificação
print(classification_report(y_test, y_pred))
