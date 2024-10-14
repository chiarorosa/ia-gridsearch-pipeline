import time

import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Gerar um dataset sintético grande (por exemplo, 1 milhão de amostras)
X, y = make_classification(n_samples=1000000, n_features=70, n_informative=25, n_classes=2, random_state=42)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Iniciar a contagem do tempo
start_time = time.time()

# Treinar o modelo XGBoost usando GPU com o máximo de parâmetros configurados
model = xgb.XGBClassifier(
    # Configurações da GPU
    tree_method="hist",  # Usar 'hist' com a configuração device
    device="cuda",  # Definir para usar a GPU
    # Configurações gerais
    booster="gbtree",  # Escolha entre 'gbtree', 'gblinear', ou 'dart'
    n_estimators=100,  # Número de árvores (ou rounds de boosting)
    max_depth=6,  # Profundidade máxima da árvore
    learning_rate=0.1,  # Taxa de aprendizado (eta)
    subsample=0.8,  # Amostras usadas para construir cada árvore (controle de overfitting)
    colsample_bytree=0.8,  # Proporção de features usadas por árvore
    colsample_bylevel=0.8,  # Proporção de features usadas em cada nível
    colsample_bynode=0.8,  # Proporção de features usadas em cada nó
    # Regularização para evitar overfitting
    reg_alpha=0.1,  # Termo de regularização L1 (Lasso)
    reg_lambda=1.0,  # Termo de regularização L2 (Ridge)
    gamma=0,  # Mínimo ganho de redução para divisão de nós (evita divisão sem ganho significativo)
    min_child_weight=1,  # Número mínimo de amostras por folha (ajuda com overfitting)
    # Configurações do método de boosting
    objective="binary:logistic",  # Tarefa de classificação binária (pode mudar para multiclass)
    eval_metric="logloss",  # Métrica de avaliação (logloss para binário)
    scale_pos_weight=1,  # Para lidar com desbalanceamento de classes
    max_delta_step=0,  # Usado para estabilidade, especialmente para classes desbalanceadas
    # Configurações para a árvore de decisão
    grow_policy="depthwise",  # Política de crescimento da árvore ('depthwise' ou 'lossguide')
    max_leaves=0,  # Número máximo de folhas em uma árvore (0 = ilimitado)
    # Otimização
    importance_type="gain",  # Tipo de importância das features: 'weight', 'gain', 'cover'
    random_state=42,  # Semente para reprodutibilidade
)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer predições
y_pred = model.predict(X_test)

# Parar a contagem do tempo
end_time = time.time()
# Calcular o tempo total
training_time = end_time - start_time
print(f"Tempo de treinamento: {training_time:.2f} segundos")

# Relatório de classificação
print(classification_report(y_test, y_pred))
