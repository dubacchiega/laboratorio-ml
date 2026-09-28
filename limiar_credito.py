import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DADOS, PROBLEMAS, SEMENTE

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
pipeline = Pipeline([("preprocessamento", preprocessamento), ("modelo", LogisticRegression(max_iter=1000))])

# Probabilidades no treino via CV, sem tocar no teste
prob_cv = cross_val_predict(pipeline, X_treino, y_treino, cv=5, method="predict_proba")[:, 1]

CUSTO_FN = 8000   # inadimplente aprovado (falso negativo)
CUSTO_FP = 1500   # bom cliente recusado (falso positivo)

def custo_total(y_real, prob, limiar):
    prev = (prob >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_real, prev).ravel()
    custo = fn * CUSTO_FN + fp * CUSTO_FP
    p = precision_score(y_real, prev, zero_division=0)
    r = recall_score(y_real, prev, zero_division=0)
    return {"limiar": limiar, "precisao": p, "recall": r, "fp": int(fp), "fn": int(fn), "custo_total": int(custo)}

limiares = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
resultados_treino = [custo_total(y_treino, prob_cv, t) for t in limiares]
for r in resultados_treino:
    print(f"limiar={r['limiar']:.2f}  precisao={r['precisao']:.3f}  recall={r['recall']:.3f}  "
          f"FP={r['fp']:4d}  FN={r['fn']:4d}  custo=R${r['custo_total']:,}")

melhor = min(resultados_treino, key=lambda r: r["custo_total"])
print(f"\nLimiar recomendado (menor custo no treino via CV): {melhor['limiar']}")

# Agora avalia esse limiar UMA vez no teste
modelo_final = pipeline.fit(X_treino, y_treino)
prob_teste = modelo_final.predict_proba(X_teste)[:, 1]
resultado_teste = custo_total(y_teste, prob_teste, melhor["limiar"])
print("\nCusto desse limiar no conjunto de TESTE:")
print(json.dumps(resultado_teste, indent=2, ensure_ascii=False))

# custo do limiar padrão 0.5 no teste, para comparação
resultado_teste_05 = custo_total(y_teste, prob_teste, 0.5)
print("\nCusto do limiar padrão 0.5 no TESTE (comparação):")
print(json.dumps(resultado_teste_05, indent=2, ensure_ascii=False))

with open("resultados_limiar.json", "w", encoding="utf-8") as f:
    json.dump({"treino_cv": resultados_treino, "limiar_recomendado": melhor,
                "teste_limiar_recomendado": resultado_teste, "teste_limiar_05": resultado_teste_05}, f,
               indent=2, ensure_ascii=False)
