#!/usr/bin/env python3
"""Consumo de eletrodomesticos por regressao linear e pseudoinversa.

Execute: python reproduzir_projeto.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

MODELOS = {
    "M0": [],
    "M1": ["T_out", "RH_out"],
    "M2": ["T_out", "RH_out", "T1", "RH_1"],
}
N_BASE_REAL = 19735
# Derivado do CSV original UCI validado, substituindo somente CRLF por LF.
SHA256_BASE_CANONICA = "e7601453ceb2f346d6d721480d74438422705ce5ac5c504c1addfac540dff413"


def sha256_canonico(conteudo: bytes) -> str:
    """Normaliza apenas CRLF para LF, sem alterar os demais bytes do CSV."""
    return hashlib.sha256(conteudo.replace(b"\r\n", b"\n")).hexdigest()


def ler_validar(csv: Path) -> tuple[pd.DataFrame, dict]:
    if not csv.is_file():
        raise FileNotFoundError(csv)
    conteudo = csv.read_bytes()
    df = pd.read_csv(io.BytesIO(conteudo), low_memory=False)
    colunas = ["date", "Appliances", "T_out", "RH_out", "T1", "RH_1"]
    ausentes = [c for c in colunas if c not in df.columns]
    if ausentes:
        raise ValueError("Colunas essenciais ausentes: " + ", ".join(ausentes))
    if len(df) != N_BASE_REAL:
        raise ValueError(f"Esperados {N_BASE_REAL} registros; encontrados {len(df)}.")
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
    sha_canonico = sha256_canonico(conteudo)
    if sha_canonico != SHA256_BASE_CANONICA:
        raise ValueError("SHA-256 divergente: use o CSV original UCI, sem alteracoes.")
    hashes = {
        "sha256": hashlib.sha256(conteudo).hexdigest(),
        "sha256_canonico": sha_canonico,
    }
    return df, hashes


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
    y_treino = treino["Appliances"].to_numpy(float)
    y_teste = teste["Appliances"].to_numpy(float)
    modelos = {}
    for nome, variaveis in MODELOS.items():
        x_treino = construir_X(treino, variaveis)
        x_teste = construir_X(teste, variaveis)
        w = np.linalg.pinv(x_treino) @ y_treino
        previsao_treino = x_treino @ w
        previsao_teste = x_teste @ w
        modelos[nome] = {
            "coeficientes": w.tolist(),
            "EQM_treino_Wh2": calcular_eqm(y_treino, previsao_treino),
            "EQM_teste_Wh2": calcular_eqm(y_teste, previsao_teste),
        }
    e0, e1, e2 = [modelos[k]["EQM_teste_Wh2"] for k in ("M0", "M1", "M2")]
    return {
        "n_treino": len(treino),
        "n_teste": len(teste),
        "modelos": modelos,
        "ganho_M2_vs_M1_pct": 100 * (e1 - e2) / e1,
        "ganho_M2_vs_M0_pct": 100 * (e0 - e2) / e0,
    }


def executar(cli=None) -> int:
    parser = argparse.ArgumentParser(description="Regressao linear por pseudoinversa na base UCI.")
    raiz = Path(__file__).resolve().parent
    parser.add_argument("--csv", type=Path, default=raiz / "dados" / "energydata_complete.csv", help="CSV original UCI")
    args = parser.parse_args(cli)
    df, _ = ler_validar(args.csv)
    resultados = analisar(df)
    print(f"Registros: {len(df)}; treino: {resultados['n_treino']}; teste: {resultados['n_teste']}")
    print(f"{'Modelo':<8} {'EQM treino (Wh2)':>22} {'EQM teste (Wh2)':>22}")
    for nome, modelo in resultados["modelos"].items():
        print(f"{nome:<8} {modelo['EQM_treino_Wh2']:22.12f} {modelo['EQM_teste_Wh2']:22.12f}")
    print("\nCoeficientes de M2:")
    for coluna, valor in zip(["intercepto", *MODELOS["M2"]], resultados["modelos"]["M2"]["coeficientes"]):
        print(f"  {coluna:<12} {valor:.12f}")
    print(f"\nReducao do EQM de M2 ante M1: {resultados['ganho_M2_vs_M1_pct']:.8f}%")
    print(f"Reducao do EQM de M2 ante M0: {resultados['ganho_M2_vs_M0_pct']:.8f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
