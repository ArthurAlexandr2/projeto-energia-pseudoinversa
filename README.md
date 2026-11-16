# Estimativa do consumo de eletrodomésticos por pseudoinversa

**Projeto Final — Álgebra Linear e Teoria da Informação, Insper**  
**Autor:** Arthur Alexandre · `arthuracc@al.insper.edu.br`

## Problema e pergunta científica

Medições de temperatura e umidade dentro de uma residência ajudam a estimar o consumo de energia dos eletrodomésticos? Em particular, investigamos se incluir sensores da **cozinha** melhora o ajuste obtido usando somente condições meteorológicas externas.

O trabalho é uma **adaptação do artigo de Candanedo, Feldheim e Deramaix (2017)**, não uma reprodução integral: mantém-se a mesma base experimental e a regressão linear múltipla, mas utiliza-se apenas a **pseudoinversa de Moore–Penrose** (conteúdo da disciplina), um conjunto menor de variáveis e uma divisão temporal do conjunto de dados.

## Referência científica e proveniência dos dados

**Artigo principal:** L. M. Candanedo, V. Feldheim e D. Deramaix, “Data driven prediction models of energy use of appliances in a low-energy house,” *Energy and Buildings*, vol. 140, pp. 81–97, 2017. DOI: https://doi.org/10.1016/j.enbuild.2017.01.083.

**Dados:** L. M. Candanedo, *Appliances Energy Prediction*, UCI Machine Learning Repository (2017). DOI: https://doi.org/10.24432/C5VC8G. Dataset: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction. A base pública foi distribuída sob **CC BY 4.0**; o arquivo `dados/energydata_complete.csv` é uma cópia do arquivo original, sem alterações, incluída com atribuição. Consulte `CREDITOS_E_METODOLOGIA.md`.

Os **19.735 registros** representam medições aproximadamente a cada 10 minutos em uma residência de baixo consumo na Bélgica, de janeiro a maio de 2016. A variável de resposta é `Appliances` (Wh por intervalo), e não o consumo de toda a casa.

## Equações e significado das matrizes

Para cada conjunto de variáveis, cada **linha de X** contém as medições de um intervalo; a primeira coluna contém 1 para estimar o termo constante. A resposta `y` guarda os consumos de eletrodomésticos, e `w` guarda os coeficientes.

```text
y ≈ Xw
w = pinv(X_treino) @ y_treino
previsao_teste = X_teste @ w
EQM = soma((y_real - y_estimado)^2) / n
```

A pseudoinversa encontra o ajuste de mínimos quadrados e funciona mesmo quando a matriz de medições possui mais linhas do que colunas. O programa verifica o posto das matrizes.

**Modelos:**

| Modelo | Colunas de X (na ordem) | Propósito |
|---|---|---|
| M0 | 1 | Referência: média do treinamento |
| M1 | 1, T_out, RH_out | Condições meteorológicas externas |
| M2 | 1, T_out, RH_out, T1, RH_1 | Inclusão de sensores da cozinha |

**Avaliação:** primeiros 15.788 registros (80%) para ajuste; últimos 3.947 (20%) para teste. Os coeficientes são calculados **somente no treino**; assim, as observações do teste não influenciam o ajuste. A divisão é cronológica, e não a validação cruzada do artigo original.

## Resultados reais obtidos

| Modelo | EQM treino (Wh²) | EQM teste (Wh²) |
|---|---:|---:|
| M0 | 11.065,98 | 8.290,73 |
| M1 | 10.791,34 | 8.369,69 |
| M2 | 10.532,01 | **8.222,81** |

Adicionar variáveis externas isoladamente (**M1**) *aumentou* o EQM de teste em **0,95%** em relação à média de treino (**M0**). A inclusão de sensores internos (**M2**) reduziu o EQM em **1,75%** relativamente a **M1**, mas apenas **0,82%** relativamente à referência simples M0. Trata-se de um ganho real porém modesto — não evidência de forte capacidade de previsão.

**Interpretação:** esses coeficientes são associações do modelo, não efeitos causais. Como os sensores e o consumo são observados no mesmo intervalo, o estudo avalia a **estimação contemporânea**, não previsão operacional de horas futuras. Resultados referem-se a uma única residência e ao período observado.

## Como reproduzir

Requer **Python 3.10+**. A partir da raiz do projeto:

```bash
python -m pip install -r requirements.txt
python reproduzir_projeto.py --csv dados/energydata_complete.csv --destino resultados
python gerar_figuras.py --csv dados/energydata_complete.csv --destino resultados/figuras
python -m unittest discover -s testes -v
```

O primeiro comando de reprodução calcula e grava `resultados/metricas_energia_reais.json` e `resultados/tabela_eqm.csv`, a partir do CSV original incluído. A leitura verifica o SHA-256, os 19.735 registros, as colunas, as datas e os valores finitos. Não há download automático; se o CSV estiver ausente, restaure o original da UCI em `dados/`.

**Relatório oficial:** o DOCX em `resultados/` é a única fonte editável; o PDF é sua versão de entrega, com uma página e duas colunas. Os scripts não geram nem sobrescrevem esses documentos. Para atualizar o texto, edite o DOCX e exporte novamente o PDF, conferindo a paginação. Word e LibreOffice não são dependências do experimento.

**Repositório do projeto:** [ArthurAlexandr2/projeto-energia-pseudoinversa](https://github.com/ArthurAlexandr2/projeto-energia-pseudoinversa). Não confundir esse endereço com o repositório dos autores da base.

O segundo script gera gráficos usando **os mesmos coeficientes da pseudoinversa**. Esses gráficos são suplementares e não substituem a tabela do relatório.

**Integridade dos dados:** SHA-256 do arquivo CSV:

```text
2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
```

## Arquivos relevantes

- `reproduzir_projeto.py`: leitura, validação, matrizes, pseudoinversa, EQM e tabela CSV.
- `gerar_figuras.py`: gráficos dos resultados reais e comparação observado/estimado.
- `dados/energydata_complete.csv`: dados originais UCI.
- `resultados/Projeto_Final_Energia_Arthur_IEEE.pdf`: relatório de uma página, duas colunas.
- `resultados/metricas_energia_reais.json`: resultados numéricos com precisão total.
- `resultados/figuras/`: figuras reproduzidas do conjunto de teste.
- `testes/test_modelos.py`: testes automáticos do cálculo de matrizes e integridade numérica.
- `CREDITOS_E_METODOLOGIA.md`: relação detalhada com o artigo e distinções metodológicas.
- `ARGUICAO_ORAL.md`: roteiro de defesa matemática e limitações.

**Aviso de escopo:** este repositório usa somente Álgebra Linear, regressão linear múltipla, pseudoinversa, operações matriciais e EQM. Não implementa nem reivindica os modelos avançados apresentados no artigo.
