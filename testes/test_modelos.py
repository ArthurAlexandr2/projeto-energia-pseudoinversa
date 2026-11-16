"""Testes pequenos de consistencia: nao sao dados do experimento cientifico."""
import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduzir_projeto import construir_X, calcular_eqm, analisar, ler_validar, separar_treino_teste, SHA256_BASE_REAL

class TestesAlgebraLinear(unittest.TestCase):
    def test_eqm_e_dimensoes_invalidas(self):
        self.assertEqual(calcular_eqm(np.array([1, 3]), np.array([0, 1])), 2.5)
        for y, previsto in [([], []), ([1, 2], [[1], [2]]), ([1], [float('inf')])]:
            with self.assertRaises(ValueError):
                calcular_eqm(np.asarray(y), np.asarray(previsto))

    def test_teste_nao_influencia_coeficientes(self):
        p = Path(__file__).resolve().parents[1] / 'dados' / 'energydata_complete.csv'
        df, meta = ler_validar(p)
        self.assertEqual(meta['sha256'], SHA256_BASE_REAL)
        self.assertEqual(len(df), 19735)
        treino, teste = separar_treino_teste(df)
        self.assertLess(treino['date'].max(), teste['date'].min())
        original = analisar(df)
        alterado = df.copy()
        alterado.loc[len(treino):, 'Appliances'] = 100000
        novo = analisar(alterado)
        for nome in original['modelos']:
            np.testing.assert_allclose(original['modelos'][nome]['coeficientes'], novo['modelos'][nome]['coeficientes'], rtol=0, atol=0)
        self.assertAlmostEqual(original['modelos']['M0']['coeficientes'][0], treino['Appliances'].mean(), places=10)

    def test_intercepto_e_dimensoes(self):
        d = pd.DataFrame({'T_out':[1, 2], 'RH_out':[4, 5]})
        x = construir_X(d, ['T_out', 'RH_out'])
        self.assertEqual(x.shape, (2,3))
        self.assertTrue(np.array_equal(x, np.array([[1,1,4],[1,2,5]],dtype=float)))

    def test_pseudoinversa_em_modelo_exato(self):
        x = np.array([[1, 0], [1, 1], [1, 2], [1, 3]],dtype=float)
        y = np.array([2,5,8,11],dtype=float)
        w = np.linalg.pinv(x) @ y
        self.assertTrue(np.allclose(w, [2,3], atol=1e-10))
        self.assertAlmostEqual(calcular_eqm(y, x @ w),0,places=20)

    def test_base_real_se_disponivel(self):
        p=Path(__file__).resolve().parents[1]/'dados'/'energydata_complete.csv'
        if not p.exists(): self.skipTest('Baixe os dados originais UCI')
        df, _ = ler_validar(p)
        result = analisar(df)
        self.assertEqual((result['n_treino'],result['n_teste']),(15788,3947))
        self.assertAlmostEqual(result['modelos']['M0']['EQM_teste_Wh2'],8290.728046428321,places=5)
        self.assertAlmostEqual(result['modelos']['M1']['EQM_teste_Wh2'],8369.693968722633,places=5)
        self.assertAlmostEqual(result['modelos']['M2']['EQM_teste_Wh2'],8222.808884263568,places=5)

if __name__ == '__main__':
    unittest.main()
