# Estimativa do Consumo de Eletrodomésticos por Regressão Linear com Pseudoinversa

**Arthur Alexandre — Álgebra Linear e Teoria da Informação, Insper**

Professor Tiago Fernandes Tavares · arthuracc@al.insper.edu.br

O projeto investiga se medições de temperatura e umidade da cozinha ajudam a estimar a energia consumida pelos eletrodomésticos de uma residência, além das condições meteorológicas externas. É uma adaptação didática de Candanedo, Feldheim e Deramaix (2017), com regressão linear múltipla resolvida por pseudoinversa.

[Relatório entregue](resultados/Projeto_Final_ArthurAlexandre.pdf) · [Repositório](https://github.com/ArthurAlexandr2/projeto-energia-pseudoinversa)

## Dados

O arquivo `dados/energydata_complete.csv` contém as **19.735 observações** da base [Appliances Energy Prediction, UCI](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), de uma residência na Bélgica, entre janeiro e maio de 2016. O alvo `Appliances` é a energia dos eletrodomésticos em **Wh por intervalo de dez minutos**, não o consumo total da casa. São usados `T_out` e `RH_out` (temperatura e umidade externas) e `T1` e `RH_1` (temperatura e umidade da cozinha), em °C e %, respectivamente.

A base é distribuída sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), com atribuição a Luis M. Candanedo. Nenhuma medição foi alterada.

## Método

Cada linha de X representa um intervalo; a primeira coluna contém 1 para o intercepto. O vetor y contém os consumos e w contém os coeficientes, que minimizam a soma dos quadrados dos resíduos:

$$\hat w=X_{\mathrm{treino}}^+y_{\mathrm{treino}},\qquad
\hat y_{\mathrm{teste}}=X_{\mathrm{teste}}\hat w,\qquad
EQM=\frac{(y-\hat y)^T(y-\hat y)}{n}.$$

| Modelo | Colunas de X, na ordem |
|---|---|
| M0 | 1: referência equivalente à média do treino |
| M1 | 1, T_out, RH_out |
| M2 | 1, T_out, RH_out, T1, RH_1 |

Os primeiros **15.788 registros (80%)** são usados no ajuste e os últimos **3.947 (20%)** no teste. O corte cronológico reserva um bloco final para avaliação sem usar seu alvo no ajuste; não demonstra independência estatística entre observações próximas. No M2, X tem 15.788 linhas e 5 colunas no treino e 3.947 linhas e 5 colunas no teste; w tem cinco coeficientes.

As variáveis externas formam uma referência simples; a cozinha foi escolhida pelo destaque de seus sensores no artigo. Não se afirma que essas quatro variáveis sejam as melhores possíveis. O artigo compara outros modelos e conjuntos de variáveis com validação cruzada; este projeto adota pseudoinversa e corte temporal 80/20. Portanto, não reproduz integralmente seu protocolo nem compara diretamente as métricas obtidas.

## Resultados

| Modelo | EQM de teste (Wh²) |
|---|---:|
| M0 | 8.290,73 |
| M1 | 8.369,69 |
| M2 | 8.222,81 |

M1 aumentou o erro de teste em **0,95%** frente a M0. M2 reduziu o erro em **1,75%** frente a M1 e em apenas **0,82%** frente a M0: a melhora sobre uma estimativa constante é modesta.

Os coeficientes são associações, não efeitos causais. Os resultados se limitam a uma residência e a um período curto; não há observação direta de ocupação ou uso de cada aparelho. Como as entradas são do mesmo intervalo do consumo, trata-se de estimação contemporânea, não previsão antecipada. O ganho observado não demonstra significância estatística.

## Como executar

Requer Python 3.10+. Na raiz do projeto, crie um ambiente com `python -m venv .venv` e ative-o com `.\.venv\Scripts\Activate.ps1` no PowerShell ou `source .venv/bin/activate` no Linux/macOS. Depois:

```bash
python -m pip install -r requirements.txt
python reproduzir_projeto.py
python -m unittest discover -s testes -v
```

O programa imprime os EQMs de treino e teste, os coeficientes de M2 e os ganhos percentuais. O CSV incluído permite executar sem download de dados. Para indicar outra cópia da mesma base, use `python reproduzir_projeto.py --csv caminho/energydata_complete.csv`.

A integridade é verificada por SHA-256 após substituir **somente CRLF por LF em memória**, sem modificar o arquivo. O hash canônico esperado é `e7601453ceb2f346d6d721480d74438422705ce5ac5c504c1addfac540dff413`, derivado do original UCI validado. Alterações em valores ou em qualquer outro conteúdo continuam sendo rejeitadas. Os testes cobrem matrizes, pseudoinversa, EQM, métricas reais, LF/CRLF e adulterações.

## Referências e autoria

1. L. M. Candanedo, V. Feldheim e D. Deramaix. “Data driven prediction models of energy use of appliances in a low-energy house.” *Energy and Buildings*, 140, 81–97, 2017. [DOI: 10.1016/j.enbuild.2017.01.083](https://doi.org/10.1016/j.enbuild.2017.01.083).
2. L. M. Candanedo. *Appliances Energy Prediction* [Dataset]. UCI Machine Learning Repository, 2017. [DOI: 10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G).

Autor: Arthur Alexandre. Projeto desenvolvido com auxílio de IA, conforme permitido na disciplina.
