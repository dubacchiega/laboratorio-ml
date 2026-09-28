<div align="center">

# 🧠 Laboratório de Aprendizado de Máquina Supervisionado

**Aula IAML 05 (Prática 2) · Inteligência Artificial e Machine Learning**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?logo=flask&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2%2B-150458?logo=pandas&logoColor=white)

*Gera dados, treina e compara modelos, avalia com rigor e serve as previsões numa página web.*

</div>

---

## 📑 Sumário

- [👥 Autores](#-autores)
- [🎯 Sobre o projeto](#-sobre-o-projeto)
- [🗂️ Estrutura](#️-estrutura)
- [🚀 Como rodar](#-como-rodar)
- [🧪 Testando a previsão de crédito](#-testando-a-previsão-de-crédito)
- [🔌 API](#-api)
- [📊 Resultados do exercício](#-resultados-do-exercício)
- [🛠️ Solução de problemas](#️-solução-de-problemas)
- [📝 Observações](#-observações)

---

## 👥 Autores

| Nome | RA |
|---|---|
| Eduardo Rafael Bacchiega | 1749416 |
| Diego Rossetto do Nascimento | 1727950 |
| Walace Alves de Araújo | 1327400 |
| Marcos da Silva Teixeira | 2158303 |

**Curso:** Ciência da Computação / Engenharia da Computação · **Professor:** Hercules Ramos · **Instituição:** FMU

---

## 🎯 Sobre o projeto

Um sistema completo de aprendizado supervisionado: os dados são gerados por código, os modelos são treinados e comparados com validação cruzada, e as previsões são servidas por uma API Flask com interface web.

O projeto tem três abas, cada uma com um problema de negócio:

| Aba | Tipo | O que prevê | Modelo escolhido |
|---|---|---|---|
| 📞 **Churn de clientes** | Classificação | Probabilidade de cancelamento do contrato | Regressão Logística |
| 🏠 **Preço de imóveis** | Regressão | Preço estimado do imóvel (R$) | Gradient Boosting |
| 💳 **Inadimplência de crédito** | Classificação | Probabilidade de o cliente não pagar o empréstimo | Regressão Logística |

> 💡 A aba de **crédito** é a do exercício prático (tópico 13 da aula).

---

## 🗂️ Estrutura

```
laboratorio-ml/
├── requirements.txt          # dependências
├── config.py                 # problemas, features e limites (fonte única de verdade)
├── gerar_dados.py            # gera os CSVs em data/
├── treinar.py                # compara, avalia e salva os modelos em models/
├── app.py                    # API Flask
├── templates/
│   └── index.html            # interface web
├── eda_credito.py            # Parte 1: análise exploratória
├── treinar_credito.py        # Partes 2 e 3: modelagem e avaliação no teste
├── limiar_credito.py         # Parte 4: limiar de decisão com custo
└── bonus_comprometimento.py  # Desafio extra: comprometimento de renda
```

---

## 🚀 Como rodar

Requer **Python 3.11 ou superior**.

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
source .venv/bin/activate         # Linux/macOS

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Subir a aplicação
python app.py
```

Na primeira execução o `app.py` gera os dados e treina os modelos sozinho. Depois é só abrir 👉 **http://127.0.0.1:5000**

<details>
<summary>Rodar os passos separadamente</summary>

```bash
python gerar_dados.py    # cria data/churn.csv, imoveis.csv e credito.csv
python treinar.py        # treina os 3 problemas e salva em models/
```

</details>

---

## 🧪 Testando a previsão de crédito

Na aba **Inadimplência de crédito**, preencha os campos e clique em **Prever**.

<table>
<tr>
<td valign="top">

**🔴 Risco alto** (esperado: ~97%)

| Campo | Valor |
|---|---|
| idade | 27 |
| renda_mensal | 2200 |
| tempo_emprego_anos | 0.5 |
| score_credito | 430 |
| dividas_ativas | 5 |
| possui_imovel | nao |
| finalidade | negocio |
| prazo_meses | 60 |
| valor_emprestimo | 45000 |

</td>
<td valign="top">

**🟢 Risco baixo** (esperado: ~0,5%)

| Campo | Valor |
|---|---|
| idade | 45 |
| renda_mensal | 12000 |
| tempo_emprego_anos | 15 |
| score_credito | 830 |
| dividas_ativas | 0 |
| possui_imovel | sim |
| finalidade | educacao |
| prazo_meses | 24 |
| valor_emprestimo | 8000 |

</td>
</tr>
</table>

Também dá para chamar a API direto pelo terminal:

```bash
curl -X POST http://127.0.0.1:5000/api/prever/credito \
  -H "Content-Type: application/json" \
  -d '{"idade": 27, "renda_mensal": 2200, "tempo_emprego_anos": 0.5, "score_credito": 430, "dividas_ativas": 5, "possui_imovel": "nao", "finalidade": "negocio", "prazo_meses": 60, "valor_emprestimo": 45000}'
```

---

## 🔌 API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Interface web |
| `GET` | `/api/problemas` | Lista problemas, features e métricas dos modelos |
| `POST` | `/api/prever/<nome>` | Retorna a previsão (`churn`, `imoveis` ou `credito`) |

---

## 📊 Resultados do exercício

**Comparação de algoritmos** (validação cruzada de 5 folds, ROC AUC):

| Algoritmo | ROC AUC |
|---|---|
| 🥇 Regressão Logística | **0,778 ± 0,022** |
| Random Forest | 0,761 ± 0,023 |
| Gradient Boosting | 0,756 ± 0,023 |
| Árvore de Decisão | 0,711 ± 0,023 |

**Avaliação no teste** (limiar 0,5):

| Acurácia | Precisão | Recall | F1 | ROC AUC |
|---|---|---|---|---|
| 80,6% | 58,9% | 25,0% | 0,351 | 0,762 |

**Limiar de decisão:** com custo de R$ 8.000 por inadimplente aprovado e R$ 1.500 por bom cliente recusado, o menor custo na validação cruzada do treino foi em **0,20**. No teste, isso reduz o custo de R$ 1.578.000 (limiar 0,5) para **R$ 1.095.000**.

---

## 📝 Observações

- Os dados são **sintéticos**, gerados com `SEMENTE = 42`, então todos obtêm os mesmos resultados. Não há dados pessoais reais.
- O conjunto de teste é usado **uma única vez**. Algoritmo e limiar são escolhidos só com validação cruzada no treino.
- Em decisões de crédito, o modelo **apoia** a decisão humana, não a substitui (LGPD, art. 20).

---

<div align="center">

*"A melhor forma de aprender é fazendo!"*

</div>
