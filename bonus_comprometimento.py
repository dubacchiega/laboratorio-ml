import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DADOS, PROBLEMAS, SEMENTE

df = pd.read_csv(DADOS / "credito.csv")
df["comprometimento_renda"] = (df["valor_emprestimo"] * 1.33 / df["prazo_meses"]) / df["renda_mensal"]

problema = PROBLEMAS["credito"]
feats_base = list(problema["features"])
categoricas = [f for f in feats_base if problema["features"][f]["tipo"] == "categoria"]
numericas_base = [f for f in feats_base if problema["features"][f]["tipo"] == "numero"]
y = df["inadimplente"]


def score(numericas):
    X = df[numericas + categoricas]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=SEMENTE, stratify=y)
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), numericas),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), categoricas),
    ])
    pipe = Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1000))])
    cv = cross_validate(pipe, X_tr, y_tr, cv=5, scoring="roc_auc")
    return cv["test_score"].mean(), cv["test_score"].std()


m1, s1 = score(numericas_base)
m2, s2 = score(numericas_base + ["comprometimento_renda"])
print(f"ROC AUC sem comprometimento_renda: {m1:.4f} ± {s1:.4f}")
print(f"ROC AUC com comprometimento_renda: {m2:.4f} ± {s2:.4f}")
