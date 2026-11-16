# Roteiro de arguição oral

Cada resposta deve vir acompanhada do entendimento de como reproduzir os cálculos no código.

**1. Qual é o problema real?** Estimar a energia usada pelos eletrodomésticos de uma casa a partir de medições ambientais contemporâneas, testando se sensores internos da cozinha melhoram o EQM.

**2. Por que escolher o artigo?** Ele estuda o mesmo problema, a mesma base e inclui regressão linear múltipla. A relevância dos sensores de cozinha no artigo motivou a comparação de M1 e M2.

**3. Nosso projeto reproduz o artigo?** Não integralmente: preserva dados e o princípio de regressão linear, mas usa somente pseudoinversa, menos variáveis e separação temporal 80/20 em vez do esquema de validação original.

**4. O que são X, y e w?** `X`: valores medidos organizados em linhas por intervalo de dez minutos, incluindo coluna inicial 1; `y`: consumo real de eletrodomésticos em Wh; `w`: coeficientes que determinam a combinação linear.

**5. Por que não usar X⁻¹?** X tem muito mais linhas que colunas, portanto não é quadrada. A pseudoinversa encontra um vetor de coeficientes que minimiza o erro quadrático dos dados de treinamento.

**6. Como implementamos o cálculo?** `w = np.linalg.pinv(X_treino) @ y_treino` e `y_estimado = X_teste @ w`. O EQM é a média do quadrado dos resíduos `y_teste - y_estimado`.

**7. Qual foi o resultado mais importante?** O EQM de teste passou de 8.369,69 em M1 para 8.222,81 em M2: ganho de 1,75%. Entretanto, frente à média de treino (M0, EQM 8.290,73), a melhora foi de apenas 0,82%.

**8. O que significa M1 piorar em relação à média?** Acrescentar variáveis pode reduzir erro no treino sem melhorar o desempenho em dados futuros. Isso não indica erro de cálculo, mas insuficiência das informações escolhidas para este recorte e período.

**9. Por que os coeficientes não demonstram causalidade?** Uma regressão observacional condiciona associações às variáveis incluídas, mas não isola causas; ocupação da casa, rotina, horários e uso efetivo dos aparelhos não são controlados pelo modelo.

**10. O que significa um coeficiente positivo para T1?** Aumentar T1 em uma unidade, mantendo as demais medições constantes na equação, aumenta o valor estimado pela magnitude desse coeficiente. Isso não garante efeito causal nem generalização.

**11. Por que não usar modelos avançados do artigo?** O objetivo é demonstrar domínio do conteúdo de Álgebra Linear ensinado, especificamente modelos lineares, pseudoinversa e EQM. Outras técnicas fogem do escopo solicitado.

**12. Quais são as limitações mais sérias?** Uma única residência, período curto, variáveis de comportamento não observadas, entradas da mesma hora da resposta e ganho modesto sobre a referência. Não se pode afirmar previsão futura sem dispor das entradas futuras.

**13. Como reproduzir?** Executar `python -m pip install -r requirements.txt`, depois `python reproduzir_projeto.py --csv dados/energydata_complete.csv --destino resultados` e `python gerar_figuras.py`. Verificar os valores em `resultados/metricas_energia_reais.json`.
