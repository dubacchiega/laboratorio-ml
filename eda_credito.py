import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv("data/credito.csv")

print("=== Shape ===", df.shape)
print("\n=== Proporção de inadimplentes ===")
print(df["inadimplente"].value_counts(normalize=True))

print("\n=== Ausentes por coluna ===")
print(df.isna().sum()[df.isna().sum() > 0])

print("\n=== Inadimplência por posse de imóvel ===")
print(df.groupby("possui_imovel")["inadimplente"].mean())

print("\n=== Inadimplência por finalidade ===")
print(df.groupby("finalidade")["inadimplente"].mean().sort_values(ascending=False))

# score em faixas
df["faixa_score"] = pd.cut(df["score_credito"], bins=[300,500,600,700,800,1000])
print("\n=== Inadimplência por faixa de score ===")
print(df.groupby("faixa_score", observed=True)["inadimplente"].mean())

# Gráfico 1: proporção de inadimplentes
fig, ax = plt.subplots(figsize=(5,4))
df["inadimplente"].value_counts(normalize=True).rename({0:"Pagou",1:"Inadimplente"}).plot(
    kind="bar", ax=ax, color=["#3b5bdb","#c92a2a"])
ax.set_title("Proporção de inadimplentes")
ax.set_ylabel("Proporção")
plt.tight_layout()
plt.savefig("report_imgs/01_proporcao.png", dpi=130)
plt.close()

# Gráfico 2: inadimplência por faixa de score
fig, ax = plt.subplots(figsize=(6,4))
df.groupby("faixa_score", observed=True)["inadimplente"].mean().plot(kind="bar", ax=ax, color="#3b5bdb")
ax.set_title("Taxa de inadimplência por faixa de score de crédito")
ax.set_ylabel("Taxa de inadimplência")
plt.tight_layout()
plt.savefig("report_imgs/02_score.png", dpi=130)
plt.close()

# Gráfico 3: inadimplência por finalidade
fig, ax = plt.subplots(figsize=(6,4))
df.groupby("finalidade")["inadimplente"].mean().sort_values(ascending=False).plot(kind="bar", ax=ax, color="#3b5bdb")
ax.set_title("Taxa de inadimplência por finalidade do empréstimo")
ax.set_ylabel("Taxa de inadimplência")
plt.tight_layout()
plt.savefig("report_imgs/03_finalidade.png", dpi=130)
plt.close()

# Gráfico 4: inadimplência por posse de imóvel
fig, ax = plt.subplots(figsize=(5,4))
df.groupby("possui_imovel")["inadimplente"].mean().plot(kind="bar", ax=ax, color="#3b5bdb")
ax.set_title("Taxa de inadimplência por posse de imóvel")
ax.set_ylabel("Taxa de inadimplência")
plt.tight_layout()
plt.savefig("report_imgs/04_imovel.png", dpi=130)
plt.close()

print("\nGráficos salvos em report_imgs/")
