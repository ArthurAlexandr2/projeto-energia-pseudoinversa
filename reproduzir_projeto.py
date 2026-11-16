#!/usr/bin/env python3
"""Projeto Final: energia de eletrodomesticos por regressao linear.

Adaptacao didatica de Candanedo, Feldheim e Deramaix (2017),
sem reproduzir seu protocolo experimental.

Execucao reproduzivel com matriz de medicoes, pseudoinversa de
Moore-Penrose e erro quadratico medio (conteudos da disciplina).

Uso: python reproduzir_projeto.py --csv dados/energydata_complete.csv --destino resultados
Saidas: metricas em JSON e tabela CSV. O relatorio oficial nao e alterado.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
MODELOS = {
    "M0": [],
    "M1": ["T_out", "RH_out"],
    "M2": ["T_out", "RH_out", "T1", "RH_1"],
}
N_BASE_REAL = 19735
SHA256_BASE_REAL = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"


def ler_validar(csv: Path) -> tuple[pd.DataFrame, dict]:
    if not csv.is_file():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv, low_memory=False)
    colunas = ["date", "Appliances", "T_out", "RH_out", "T1", "RH_1"]
    ausentes = [c for c in colunas if c not in df.columns]
    if ausentes:
        raise ValueError("Colunas essenciais ausentes: " + ", ".join(ausentes))
    if len(df) != N_BASE_REAL:
        raise ValueError(
            f"A base oficial completa tem {N_BASE_REAL} linhas; arquivo recebido tem {len(df)}. "
            "Nao vou usar uma base parcial ou simulada como se fosse a original."
        )
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for c in colunas[1:]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if df[colunas].isna().any().any():
        raise ValueError("Base contem valores ausentes nas colunas utilizadas; inspecionar antes de prosseguir.")
    if not np.isfinite(df[colunas[1:]].to_numpy(float)).all():
        raise ValueError("Base contem valores numericos nao finitos.")
    if (df["Appliances"] < 0).any():
        raise ValueError("O consumo energetico nao deveria ser negativo.")
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError("Datas duplicadas no CSV.")
    sha = hashlib.sha256(csv.read_bytes()).hexdigest()
    if sha != SHA256_BASE_REAL:
        raise ValueError("SHA-256 divergente: use o CSV original UCI, sem alteracoes.")
    meta = {
        "fonte": "UCI Machine Learning Repository, Appliances Energy Prediction (id=374)",
        "csv": csv.name,
        "sha256": sha,
        "n_observacoes": int(len(df)),
        "inicio": str(df["date"].iloc[0]),
        "fim": str(df["date"].iloc[-1]),
        "variavel_alvo": "Appliances (Wh por intervalo de 10 minutos)",
    }
    return df, meta


def construir_X(df: pd.DataFrame, variaveis: list[str]) -> np.ndarray:
    """Intercepto + variaveis medidas em colunas; metodo das aulas."""
    x = np.ones((len(df), 1 + len(variaveis)), dtype=float)
    if variaveis:
        x[:, 1:] = df[variaveis].to_numpy(float)
    return x


def calcular_eqm(y: np.ndarray, previsto: np.ndarray) -> float:
    """EQM para dois vetores de mesma dimensao, sem broadcasting implicito."""
    y = np.asarray(y, dtype=float)
    previsto = np.asarray(previsto, dtype=float)
    if y.ndim != 1 or previsto.shape != y.shape or y.size == 0:
        raise ValueError("O EQM requer vetores unidimensionais, nao vazios e de mesmo tamanho.")
    if not (np.isfinite(y).all() and np.isfinite(previsto).all()):
        raise ValueError("O EQM requer valores finitos.")
    e = y - previsto
    return float((e.T @ e) / len(e))


def separar_treino_teste(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primeiros 80% para ajuste; ultimos 20% para avaliacao."""
    if len(df) < 2:
        raise ValueError("Sao necessarias observacoes de treino e de teste.")
    if df["date"].isna().any() or not df["date"].is_monotonic_increasing or df["date"].duplicated().any():
        raise ValueError("As datas devem ser validas, unicas e estar em ordem cronologica.")
    corte = int(0.8 * len(df))
    return df.iloc[:corte], df.iloc[corte:]


def analisar(df: pd.DataFrame) -> dict:
    treino, teste = separar_treino_teste(df)
    yt = treino["Appliances"].to_numpy(float)
    yv = teste["Appliances"].to_numpy(float)
    modelos = {}
    for nome, vars_ in MODELOS.items():
        xt = construir_X(treino, vars_)
        xv = construir_X(teste, vars_)
        w = np.linalg.pinv(xt) @ yt
        modelos[nome] = {
            "colunas_X": ["intercepto", *vars_],
            "coeficientes": [float(z) for z in w],
            "n_coeficientes": int(w.size),
            "posto_X_treino": int(np.linalg.matrix_rank(xt)),
            "EQM_treino_Wh2": calcular_eqm(yt, xt @ w),
            "EQM_teste_Wh2": calcular_eqm(yv, xv @ w),
            "previsoes_negativas_teste": int(np.sum((xv @ w) < 0)),
        }
    e0, e1, e2 = [modelos[k]["EQM_teste_Wh2"] for k in ("M0", "M1", "M2")]
    return {
        "n_treino": len(treino),
        "n_teste": len(teste),
        "inicio_teste": str(teste["date"].iloc[0]),
        "modelos": modelos,
        "ganho_M1_vs_M0_pct": 100 * (e0 - e1) / e0,
        "ganho_M2_vs_M1_pct": 100 * (e1 - e2) / e1,
        "melhor_EQM_teste": min(modelos, key=lambda k: modelos[k]["EQM_teste_Wh2"]),
    }


def executar(cli=None) -> int:
    parser = argparse.ArgumentParser(description="Reproduzir metricas e tabela por pseudoinversa, preservando o relatorio oficial.")
    raiz = Path(__file__).resolve().parent
    parser.add_argument("--csv", type=Path, default=raiz / "dados" / "energydata_complete.csv", help="CSV original UCI")
    parser.add_argument("--destino", type=Path, default=raiz / "resultados", help="Pasta para JSON e tabela CSV")
    args = parser.parse_args(cli)
    df, metadados = ler_validar(args.csv)
    resultados = analisar(df)
    info = {"dados": metadados, "avaliacao": resultados, "metodologia": "pseudoinversa Moore-Penrose; modelos predefinidos; separacao cronologica 80/20; EQM"}
    # Somente arquivos numericos sao gravados. DOCX/PDF nunca sao abertos para escrita.
    args.destino.mkdir(parents=True, exist_ok=True)
    jsonpath = args.destino / "metricas_energia_reais.json"
    jsonpath.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tabela = pd.DataFrame([
        {"modelo": nome, "n_coeficientes": valores["n_coeficientes"],
         "EQM_treino_Wh2": valores["EQM_treino_Wh2"], "EQM_teste_Wh2": valores["EQM_teste_Wh2"]}
        for nome, valores in resultados["modelos"].items()
    ])
    csvpath = args.destino / "tabela_eqm.csv"
    tabela.to_csv(csvpath, index=False, encoding="utf-8", lineterminator="\n")
    print(tabela.to_string(index=False))
    print(f"Treino: {resultados['n_treino']}; teste: {resultados['n_teste']}")
    print(f"Reducao M2/M1: {resultados['ganho_M2_vs_M1_pct']:.8f}%")
    print("JSON:", jsonpath)
    print("Tabela:", csvpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
