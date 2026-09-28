const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageOrientation
} = require("docx");

const IMG = (name, w, h) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 120 },
  children: [new ImageRun({ data: fs.readFileSync(`report_imgs/${name}.png`), transformation: { width: w, height: h }, type: "png" })],
});

const H1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
const H2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
const P = (t, opts = {}) => new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: t, ...opts })] });
const Bold = (t) => new TextRun({ text: t, bold: true });
const Bullet = (t) => new Paragraph({ text: t, bullet: { level: 0 }, spacing: { after: 60 } });

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1D2330" } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: !!opts.header, color: opts.header ? "FFFFFF" : undefined })] })],
  });
}

function table(headers, rows, widths) {
  const colW = widths || headers.map(() => Math.floor(9000 / headers.length));
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: colW,
    rows: [
      new TableRow({ children: headers.map((h, i) => cell(h, { header: true, width: colW[i] })) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, { width: colW[i] })) })),
    ],
  });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } },
    children: [
      new Paragraph({ text: "Exercício Prático — Aula IAML 05", heading: HeadingLevel.TITLE, spacing: { after: 60 } }),
      new Paragraph({ text: "Previsão de inadimplência de crédito (aprendizado supervisionado)", heading: HeadingLevel.HEADING_3, spacing: { after: 200 } }),
      P("Eduardo — Ciência da Computação / Engenharia da Computação", { italics: true }),
      P("Base: data/credito.csv gerado por gerar_dados.py — 6.000 contratos, 21,03% inadimplentes.", { italics: true }),

      H1("Parte 1 — Análise exploratória"),
      P("O conjunto tem 6.000 linhas e 11 colunas. A proporção de inadimplentes é de 21,03% (78,97% pagaram), ou seja, as classes são moderadamente desbalanceadas — um modelo que sempre previsse \"pagou\" já acertaria quase 79% das vezes, então a acurácia isolada não é uma boa métrica aqui (mesmo raciocínio da seção 9.1 do material)."),
      IMG("01_proporcao", 320, 256),

      H2("Valores ausentes"),
      P("Duas colunas têm ausentes, propositalmente inseridos pelo gerador: renda_mensal (231 linhas, 3,85%) e tempo_emprego_anos (183 linhas, 3,05%). Nenhuma outra coluna tem ausentes. Como a proporção é pequena, a imputação pela mediana (robusta a outliers de renda) é suficiente e será feita dentro do Pipeline, sem vazamento."),

      H2("Inadimplência por score de crédito"),
      P("Há uma relação forte e monotônica: score entre 300–500 tem 47,4% de inadimplência, caindo para 30,8% (500–600), 16,8% (600–700), 10,2% (700–800) e apenas 1,8% acima de 800. O score de crédito é, isoladamente, o sinal mais forte do conjunto."),
      IMG("02_score", 384, 256),

      H2("Inadimplência por finalidade e por posse de imóvel"),
      P("Empréstimos para \"negócio\" têm a maior taxa de inadimplência (27,7%), acima da média geral; as demais finalidades ficam entre 18,9% e 20,4%. Quem possui imóvel próprio inadimple menos (18,9%) do que quem não possui (22,5%) — provavelmente por servir como uma proxy de estabilidade financeira."),
      IMG("03_finalidade", 384, 256),
      IMG("04_imovel", 320, 256),

      H2("Colunas a descartar"),
      P("id_contrato é apenas um identificador sequencial sem relação causal com o pagamento — deve ficar fora das features, exatamente como alertado na seção 5 do material (risco de vazamento/decoreba). Todas as demais colunas (idade, renda, score, dívidas, posse de imóvel, finalidade, prazo e valor do empréstimo) têm relação plausível com a capacidade de pagamento e foram mantidas."),

      H1("Parte 2 — Modelagem"),
      P("Os dados foram divididos em 80% treino / 20% teste, estratificados por inadimplente, com random_state=42. O pré-processamento (imputação de mediana + padronização nas numéricas; imputação de moda + one-hot nas categóricas) foi encapsulado em um ColumnTransformer dentro de um Pipeline, de forma que as estatísticas de imputação e escala são aprendidas apenas no treino — sem vazamento. Quatro algoritmos foram comparados por validação cruzada de 5 folds no treino, usando ROC AUC como métrica (adequada por não depender de um limiar fixo e por lidar bem com classes desbalanceadas):"),
      table(
        ["Algoritmo", "ROC AUC (média ± desvio, 5-fold CV)"],
        [
          ["Regressão Logística", "0,7781 ± 0,0224"],
          ["Árvore de Decisão", "0,7106 ± 0,0232"],
          ["Random Forest", "0,7606 ± 0,0228"],
          ["Gradient Boosting", "0,7547 ± 0,0217"],
        ],
        [4500, 4500]
      ),
      P(""),
      P("A Regressão Logística venceu, com uma leitura análoga à do churn no material: os principais sinais (score de crédito, comprometimento de renda embutido no valor/prazo/renda) têm relação aproximadamente monotônica/linear com a chance de inadimplência, então modelos mais flexíveis (árvores, ensembles) apenas adicionam variância sem capturar interações fortes o bastante para compensar. O modelo escolhido foi treinado no treino completo e salvo em models/credito.joblib."),

      H1("Parte 3 — Avaliação no teste"),
      P("Avaliando o modelo escolhido (Regressão Logística) uma única vez no conjunto de teste (1.200 contratos), com limiar padrão 0,5:"),
      table(
        ["Métrica", "Valor"],
        [
          ["Acurácia", "80,6%"],
          ["Precisão", "58,9%"],
          ["Recall", "25,0%"],
          ["F1", "0,351"],
          ["ROC AUC", "0,762"],
          ["Matriz de confusão (VN, FP, FN, VP)", "904, 44, 189, 63"],
        ],
        [4500, 4500]
      ),
      P(""),
      IMG("05_roc", 300, 300),
      H2("Falso positivo x falso negativo neste negócio"),
      P("Falso positivo (FP): o modelo prevê inadimplência, mas o cliente pagaria — um bom cliente é recusado, e a fintech deixa de lucrar com aquele contrato. Falso negativo (FN): o modelo prevê que o cliente pagaria, mas ele se torna inadimplente — o crédito é aprovado e o valor é perdido."),
      P("Neste negócio o falso negativo é claramente mais caro: com o limiar padrão 0,5 o recall é de apenas 25% — o modelo deixa passar 189 dos 252 inadimplentes reais do teste (75% dos casos de risco!). Isso porque o limiar de 0,5 favorece a precisão em detrimento do recall, o que é uma escolha ruim quando perder um inadimplente custa muito mais do que recusar um bom pagador (ver Parte 4)."),

      H1("Parte 4 — Limiar de decisão"),
      P("As probabilidades foram obtidas sem tocar no teste, via cross_val_predict(pipeline, X_treino, y_treino, cv=5, method=\"predict_proba\"). Usando o custo de negócio informado (FN = R$ 8.000 por inadimplente aprovado; FP = R$ 1.500 por bom cliente recusado), o custo total (FN×8.000 + FP×1.500) foi calculado no treino para oito limiares:"),
      table(
        ["Limiar", "Precisão", "Recall", "FP", "FN", "Custo total (treino, CV)"],
        [
          ["0,10", "28,7%", "91,2%", "2.287", "89", "R$ 4.142.500"],
          ["0,15", "33,3%", "82,2%", "1.660", "180", "R$ 3.930.000"],
          ["0,20", "38,8%", "74,3%", "1.185", "260", "R$ 3.857.500"],
          ["0,25", "42,8%", "63,8%", "860", "366", "R$ 4.218.000"],
          ["0,30", "47,2%", "55,0%", "620", "455", "R$ 4.570.000"],
          ["0,35", "50,5%", "45,9%", "455", "546", "R$ 5.050.500"],
          ["0,40", "53,4%", "37,9%", "334", "627", "R$ 5.517.000"],
          ["0,50", "60,9%", "24,1%", "156", "767", "R$ 6.370.000"],
        ],
        [1100, 1300, 1300, 1300, 1300, 2700]
      ),
      P(""),
      P("O menor custo no treino ocorre no limiar 0,20 (R$ 3.857.500), bem abaixo do custo com o limiar padrão 0,50 (R$ 6.370.000) — uma economia de quase 40%. Isso confirma a leitura da Parte 3: como o FN custa mais de 5× o FP, vale a pena aceitar mais falsos positivos (recusar mais bons clientes) para capturar mais inadimplentes de verdade."),
      P([Bold("Limiar recomendado: 0,20.")].map(r => r.text).join(""), {}),
      P("Aplicando esse limiar uma única vez no conjunto de teste: precisão 35,7%, recall 69,0%, 314 FP, 78 FN, custo total de R$ 1.095.000 — contra R$ 1.578.000 com o limiar padrão 0,5 no mesmo teste (redução de ~31% no custo)."),

      H1("Parte 5 — Integração ao sistema"),
      P("O problema credito foi adicionado ao dicionário PROBLEMAS de config.py, com as 9 features (numéricas e categóricas), seus limites e opções, no mesmo padrão de churn e imoveis. Como app.py e templates/index.html do material já são genéricos — ambos iteram sobre PROBLEMAS via /api/problemas em vez de terem qualquer referência fixa a churn ou imóveis — nenhuma alteração de código foi necessária além do config.py: ao rodar python treinar.py, uma terceira aba \"Inadimplência de crédito\" passa a aparecer automaticamente na interface, com seu próprio formulário e métricas."),
      P("Testes via curl confirmaram o funcionamento (POST /api/prever/credito):"),
      Bullet("Caso de risco alto (score 430, sem imóvel, dívidas ativas, empréstimo alto para negócio): probabilidade de inadimplência = 97,4%."),
      Bullet("Caso de risco baixo (score 830, imóvel próprio, sem dívidas, empréstimo modesto para educação): probabilidade de inadimplência = 0,5%."),
      P("Como a execução do Flask é local (python app.py, http://localhost:5000), as capturas de tela da aba \"Inadimplência de crédito\" em funcionamento devem ser feitas rodando o projeto na sua máquina — os arquivos completos (config.py, gerar_dados.py, treinar.py, app.py, templates/index.html) estão prontos e testados para isso."),

      H1("Parte 6 — Reflexão ética"),
      P("Um modelo de crédito aprende exclusivamente com o passado: se historicamente certos grupos tiveram crédito negado por preconceito (explícito ou estrutural), essas decisões entram como \"exemplos corretos\" nos dados de treino, e o modelo aprende a reproduzir — e a automatizar em escala — a mesma discriminação, mesmo sem nenhuma feature de raça ou gênero, porque variáveis aparentemente neutras (CEP, finalidade do empréstimo, tipo de emprego) podem funcionar como proxies dessas características. Por isso, mesmo que melhorassem a métrica de validação, eu não usaria features sensíveis diretas (gênero, raça, religião, orientação sexual) nem substitutas evidentes de local de moradia muito granulares (CEP completo), já que no Brasil elas tendem a correlacionar-se fortemente com renda e composição racial da vizinhança. A LGPD reforça esse cuidado: exige finalidade específica e dados mínimos para o tratamento, e o art. 20 garante ao titular o direito de solicitar revisão de decisões tomadas unicamente por tratamento automatizado — o que, na prática, significa que um modelo de crédito como este deve apoiar a decisão de um analista humano (\"humano no circuito\"), e não substituí-lo, especialmente nos casos limítrofes em torno do limiar escolhido."),

      H1("Desafio extra — feature de comprometimento de renda"),
      P("Foi criada a feature comprometimento_renda = (valor_emprestimo × 1,33 / prazo_meses) / renda_mensal e comparado o ROC AUC (5-fold CV) da Regressão Logística com e sem ela:"),
      table(
        ["Configuração", "ROC AUC (média ± desvio)"],
        [
          ["Sem comprometimento_renda", "0,7781 ± 0,0224"],
          ["Com comprometimento_renda", "0,7779 ± 0,0229"],
        ],
        [4500, 4500]
      ),
      P(""),
      P("Praticamente nenhuma mudança — o que faz sentido: a Regressão Logística é um modelo linear, e comprometimento_renda é uma razão (uma interação não linear) entre três variáveis que o modelo já recebe separadamente (valor_emprestimo, prazo_meses, renda_mensal). Um modelo linear não consegue reconstruir sozinho essa razão a partir das variáveis originais, então fornecer a razão pronta ajudaria — mas nesse caso específico o ganho foi nulo, sugerindo que o sinal de comprometimento já está sendo parcialmente capturado por outra via (o score_credito, por exemplo, tende a refletir o histórico de comprometimento do cliente). Esse resultado ilustra bem o ponto do material: entender o porquê de um resultado vale mais do que só comparar números."),

      H1("Arquivos do projeto"),
      P("laboratorio-ml/ contém: requirements.txt, config.py (com os 3 problemas), gerar_dados.py, treinar.py, app.py, templates/index.html, além dos scripts auxiliares usados neste relatório (eda_credito.py, treinar_credito.py, limiar_credito.py, bonus_comprometimento.py). Para rodar: pip install -r requirements.txt seguido de python app.py, e acessar http://localhost:5000."),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => fs.writeFileSync("relatorio_exercicio_credito.docx", buf));
