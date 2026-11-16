# Fundamentação científica e adaptação da metodologia

## O que pertence à referência científica

O artigo [Candanedo, Feldheim e Deramaix (2017)](https://doi.org/10.1016/j.enbuild.2017.01.083) investiga como estimar o consumo agregado dos eletrodomésticos de uma residência de baixo consumo de energia. Usa dados obtidos de sensores de temperatura e umidade em diferentes ambientes e de uma estação meteorológica. Os autores comparam **regressão linear múltipla** com outros métodos mais avançados, utilizando diferentes subconjuntos de variáveis, validação cruzada e avaliação com dados de teste.

O artigo dá destaque aos sensores de **cozinha, lavanderia e sala**, e aponta limitações na descrição linear do consumo. Esses achados motivam nossa seleção de `T1` e `RH_1` da cozinha, sem pretender replicar todas as escolhas e resultados dos autores.

## O que foi deliberadamente adaptado

| Aspecto | Pesquisa original | Este projeto |
|---|---|---|
| Problema | Consumo agregado de eletrodomésticos | Mesmo problema, com pergunta restrita sobre sensores da cozinha |
| Dataset | Medições da residência e condições ambientais | Mesmo CSV público, preservado integralmente |
| Métodos | Regressão linear + métodos avançados | Somente regressão linear, com `np.linalg.pinv` |
| Conjunto de variáveis | Muitos sensores, tempo e clima | T_out, RH_out, T1 e RH_1, além do intercepto |
| Estratégia de comparação | Diferentes modelos e subconjuntos | M0, M1, M2, sucessivamente mais informativos |
| Avaliação | Validação cruzada e conjunto de teste | Partição cronológica fixa 80/20, sem vazamento do teste |
| Métricas | Entre outras, RMSE e R² | EQM, empregado na disciplina |

A comparação é **qualitativa, não numérica direta**: diferenças nos preditores, nos métodos de validação e nas métricas impedem interpretar nosso EQM como reprodução dos valores relatados no artigo.

## Equacionamento conectado ao problema

Denote por `y_i` a energia consumida pelos eletrodomésticos durante o intervalo `i`, em Wh, e por `X[i,j]` a j-ésima medição ambiental observada nesse intervalo. O modelo M2 tem a forma:

`y_estimado_i = w0 + w1*T_out_i + w2*RH_out_i + w3*T1_i + w4*RH_1_i`.

Se `n` é a quantidade de intervalos e `p` é o número de coeficientes, `X` tem `n` linhas e `p` colunas; `y` tem `n` elementos. A pseudoinversa de Moore–Penrose estima os coeficientes minimizando o quadrado dos resíduos, isto é, a soma dos quadrados de `y - Xw`. No teste, aplicamos o vetor estimado sem recalcular seus coeficientes.

## Decisões experimentais e limites

`T_out` e `RH_out` fornecem uma referência meteorológica simples; `T1` e `RH_1` acrescentam as mesmas grandezas medidas na cozinha. A seleção é um recorte didático, não uma alegação de que sejam os melhores preditores do artigo. A pseudoinversa foi escolhida para resolver mínimos quadrados diretamente com o conteúdo da disciplina.

A proporção 80/20 reserva a maior parte das observações ao ajuste e um bloco final contínuo ao teste, sem otimizar o corte. A ausência de sobreposição entre linhas não demonstra independência estatística entre medições próximas. O ganho observado não constitui evidência de significância estatística.

O artigo original foi conferido na [cópia institucional da Universidade de Mons](https://docnum.umons.ac.be/Apps/webrequest/request.aspx?DocGuid=727540F3-141B-46DF-87D9-F1D7AC421E02&DomaineGuid=c7b423fd-d183-486c-9cec-966066b9b364&Filename=1-s2.0-S0378778816308970-main.pdf): compara métodos com validação cruzada de dez partes e três repetições, além de avaliação em teste separado. Esse protocolo difere do adotado aqui.

1. **M0 como referência:** prever sempre a média do treinamento testa se acrescentar variáveis produz ganho efetivo.
2. **M1 versus M2:** isola a contribuição adicional da cozinha nas condições deste modelo.
3. **80/20 cronológico:** respeita a sequência temporal e evita utilizar observações futuras no ajuste; não é o mesmo protocolo de validação cruzada do artigo.
4. **EQM de teste:** privilegia comparações sobre os mesmos dados não utilizados no treinamento.
5. **Interpretação restrita:** não há inferência causal, previsão antecipada sem dados futuros ou generalização para outras residências.
6. **Cuidado com os coeficientes:** a presença simultânea de variáveis ambientais correlacionadas pode alterar os sinais de um coeficiente. Um sinal negativo não prova um efeito físico negativo.

## Integridade dos dados e licença

O arquivo `dados/energydata_complete.csv` é uma cópia do conjunto [Appliances Energy Prediction, UCI](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), disponibilizado sob licença **Creative Commons Attribution 4.0 International (CC BY 4.0)**. A redistribuição deve preservar crédito a Luis M. Candanedo e a fonte original. Nenhuma observação da base foi inventada ou modificada.

**Referência bibliográfica:** L. M. Candanedo, V. Feldheim e D. Deramaix, “Data driven prediction models of energy use of appliances in a low-energy house,” *Energy and Buildings*, 140, 81–97, 2017. https://doi.org/10.1016/j.enbuild.2017.01.083.

**Referência do dataset:** L. M. Candanedo, *Appliances Energy Prediction*, UCI Machine Learning Repository, 2017. https://doi.org/10.24432/C5VC8G.
