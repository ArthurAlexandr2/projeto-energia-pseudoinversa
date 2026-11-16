"""Graficos reproduziveis com os dados reais da UCI e pseudoinversa.

Apenas visualizacao com matplotlib; regressao sempre por np.linalg.pinv.
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reproduzir_projeto import MODELOS, analisar, construir_X, ler_validar, separar_treino_teste


def criar_figuras(csv: Path, pasta: Path):
    df, _ = ler_validar(csv)
    resultados = analisar(df)
    _, te = separar_treino_teste(df)
    yte = te['Appliances'].to_numpy(float)
    previsoes = {}
    eqm_teste = {}
    for nome, colunas in MODELOS.items():
        xv = construir_X(te, colunas)
        modelo = resultados['modelos'][nome]
        w = modelo['coeficientes']
        previsoes[nome] = xv @ w
        eqm_teste[nome] = modelo['EQM_teste_Wh2']
    pasta.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=150)
    nomes = list(eqm_teste)
    valores = [eqm_teste[k] for k in nomes]
    bars = ax.bar(nomes, valores, width=0.5)
    for b, v in zip(bars, valores):
        ax.text(b.get_x() + b.get_width() / 2, v + 70, f'{v:,.1f}'.replace(',', 'X').replace('.', ',').replace('X', '.'), ha='center', fontsize=10)
    ax.set_ylim(0, max(valores) * 1.13)
    ax.set_ylabel('EQM no teste (Wh²)')
    ax.set_xlabel('Modelo')
    ax.set_title('Comparação em observações de teste não usadas no ajuste')
    fig.tight_layout()
    fig.savefig(pasta / 'comparacao_eqm_teste.png', dpi=170)
    plt.close(fig)

    # Um dia do teste. Visualizacao, nao avaliacao adicional ou treino.
    passos = min(144, len(te))
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    t = pd.to_datetime(te['date'].iloc[:passos])
    ax.plot(t, yte[:passos], label='Consumo medido', linewidth=1.3)
    ax.plot(t, previsoes['M2'][:passos], label='Estimativa M2', linewidth=1.5)
    ax.set_ylabel('Consumo em Wh / 10 min')
    ax.set_xlabel('Data e hora')
    ax.set_title('Exemplo de valores reais versus estimados (primeiras 24 h do teste)')
    ax.legend(loc='upper right')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(pasta / 'observado_vs_estimado_M2.png', dpi=170)
    plt.close(fig)
    print('Figuras geradas em', pasta)
    for n,v in eqm_teste.items():
        print(f'{n}: EQM teste = {v:.4f} Wh²')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Figuras com as medicoes reais e o modelo M2.')
    parser.add_argument('--csv', type=Path, default=Path(__file__).resolve().parent / 'dados' / 'energydata_complete.csv')
    parser.add_argument('--destino', type=Path, default=Path(__file__).resolve().parent / 'resultados' / 'figuras')
    a = parser.parse_args()
    criar_figuras(a.csv, a.destino)
