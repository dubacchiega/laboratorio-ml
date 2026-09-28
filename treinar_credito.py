import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import DADOS, MODELOS, PROBLEMAS, SEMENTE

problema = PROBLEMAS["credito"]
df = pd.read_csv(DADOS / "credito.csv")
X = df[list(problema["features"])]
y = df[problema["alvo"]]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=.2, random_state=SEMENTE, stratify=y)

features = problema["features"]
numericas = [f for f in features if features[f]["tipo"] == "numero"]
categoricas = [f for f in features if features[f]["tipo"] == "categoria"]

preprocessamento = ColumnTransformer([
    ("numericas", Pipeline([("preencher", SimpleImputer(strategy="median")),
                             ("padronizar", StandardScaler())]), numericas),
    ("categoricas", Pipeline([("preencher", SimpleImputer(strategy="most_frequent")),
                               ("one_hot", OneHotEncoder(handle_unknown="ignore"))]), categoricas),
])

candidatos = {
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "Árvore de Decisão": DecisionTreeClassifier(max_depth=5, random_state=SEMENTE),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=SEMENTE),
    "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=.05, random_state=SEMENTE),
}

def pipe(modelo):
    return Pipeline([("preprocessamento", preprocessamento), ("modelo", modelo)])

print("=== Parte 2: Validação cruzada (5 folds, ROC AUC) ===")
comparacao = {}
for nome, modelo in candidatos.items():
    cv = cross_validate(pipe(modelo), X_treino, y_treino, cv=5, scoring="roc_auc")
    notas = cv["test_score"]
    comparacao[nome] = {"media": float(notas.mean()), "desvio": float(notas.std())}
    print(f"  {nome:<22} {notas.mean():.4f} ± {notas.std():.4f}")

escolhido = max(comparacao, key=lambda a: comparacao[a]["media"])
print(f"\nEscolhido: {escolhido}")

modelo_final = pipe(candidatos[escolhido]).fit(X_treino, y_treino)

print("\n=== Parte 3: Avaliação no teste ===")
prob_teste = modelo_final.predict_proba(X_teste)[:, 1]
prev_teste = (prob_teste >= .5).astype(int)
matriz = confusion_matrix(y_teste, prev_teste)
teste_metricas = {
    "acuracia": accuracy_score(y_teste, prev_teste),
    "precisao": precision_score(y_teste, prev_teste),
    "recall": recall_score(y_teste, prev_teste),
    "f1": f1_score(y_teste, prev_teste),
    "roc_auc": roc_auc_score(y_teste, prob_teste),
    "matriz_confusao": matriz.tolist(),
}
print(json.dumps(teste_metricas, indent=2, ensure_ascii=False))

# Curva ROC
fpr, tpr, _ = roc_curve(y_teste, prob_teste)
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(fpr, tpr, color="#3b5bdb", label=f"ROC AUC = {teste_metricas['roc_auc']:.3f}")
ax.plot([0,1],[0,1],"--",color="gray")
ax.set_xlabel("Taxa de falso positivo")
ax.set_ylabel("Taxa de verdadeiro positivo")
ax.set_title("Curva ROC — modelo de crédito (teste)")
ax.legend()
plt.tight_layout()
plt.savefig("report_imgs/05_roc.png", dpi=130)
plt.close()

MODELOS.mkdir(exist_ok=True)
joblib.dump(modelo_final, MODELOS / "credito.joblib", compress=3)

with open("resultados_credito.json", "w", encoding="utf-8") as f:
    json.dump({"comparacao": comparacao, "escolhido": escolhido, "teste": teste_metricas}, f, indent=2, ensure_ascii=False)

print("\nSalvo: models/credito.joblib, resultados_credito.json, report_imgs/05_roc.png")
