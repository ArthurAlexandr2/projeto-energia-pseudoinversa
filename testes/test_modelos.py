"""Testes pequenos de consistencia: nao sao dados do experimento cientifico."""
import unittest
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduzir_projeto import construir_X, calcular_eqm, analisar, ler_validar, separar_treino_teste, SHA256_BASE_CANONICA, sha256_canonico

class TestesAlgebraLinear(unittest.TestCase):
    def test_eqm_e_dimensoes_invalidas(self):
        self.assertEqual(calcular_eqm(np.array([1, 3]), np.array([0, 1])), 2.5)
        for y, previsto in [([], []), ([1, 2], [[1], [2]]), ([1], [float('inf')])]:
            with self.assertRaises(ValueError):
                calcular_eqm(np.asarray(y), np.asarray(previsto))

    def test_teste_nao_influencia_coeficientes(self):
        p = Path(__file__).resolve().parents[1] / 'dados' / 'energydata_complete.csv'
        df, meta = ler_validar(p)
        self.assertEqual(meta['sha256_canonico'], SHA256_BASE_CANONICA)
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


class TestesIntegridadeCSV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        origem = Path(__file__).resolve().parents[1] / 'dados' / 'energydata_complete.csv'
        cls.lf = origem.read_bytes().replace(b'\r\n', b'\n')
        cls.crlf = cls.lf.replace(b'\n', b'\r\n')

    def test_base_lf_e_crlf(self):
        with TemporaryDirectory() as pasta:
            quadros = []
            resultados = []
            for nome, conteudo in [('lf', self.lf), ('crlf', self.crlf)]:
                with self.subTest(formato=nome):
                    caminho = Path(pasta) / f'{nome}.csv'
                    caminho.write_bytes(conteudo)
                    df, meta = ler_validar(caminho)
                    self.assertEqual(meta['sha256'], hashlib.sha256(conteudo).hexdigest())
                    self.assertEqual(meta['sha256_canonico'], SHA256_BASE_CANONICA)
                    self.assertEqual(len(df), 19735)
                    quadros.append(df)
                    resultados.append(analisar(df))
            pd.testing.assert_frame_equal(*quadros)
            self.assertEqual(*resultados)

    def test_alteracao_de_valor_rejeitada(self):
        linhas = self.lf.splitlines(keepends=True)
        campos = linhas[1].split(b',')
        valor = campos[1].strip(b'" ')
        campos[1] = campos[1].replace(valor, str(int(valor) + 10).encode('ascii'), 1)
        linhas[1] = b','.join(campos)
        alterado = b''.join(linhas)
        self.assertNotEqual(sha256_canonico(alterado), SHA256_BASE_CANONICA)
        with TemporaryDirectory() as pasta:
            for nome, conteudo in [('lf', alterado), ('crlf', alterado.replace(b'\n', b'\r\n'))]:
                with self.subTest(formato=nome):
                    caminho = Path(pasta) / f'{nome}.csv'
                    caminho.write_bytes(conteudo)
                    with self.assertRaisesRegex(ValueError, 'SHA-256 divergente'):
                        ler_validar(caminho)

    def test_alteracoes_de_estrutura_e_conteudo_rejeitadas(self):
        linhas = self.lf.splitlines(keepends=True)
        alteracoes = {
            'coluna': self.lf.replace(b'"Appliances"', b'"Appliances_alterado"', 1),
            'coluna_extra': b'extra,' + linhas[0] + b''.join(b'0,' + linha for linha in linhas[1:]),
            'linha_removida': b''.join(linhas[:-1]),
            'ordem_das_linhas': b''.join([linhas[0], linhas[2], linhas[1], *linhas[3:]]),
            'espaco_extra': self.lf + b' ',
            'sem_nova_linha_final': self.lf[:-1],
            'bom_utf8': b'\xef\xbb\xbf' + self.lf,
            'retorno_isolado': self.lf.replace(b'\n', b'\r'),
        }
        with TemporaryDirectory() as pasta:
            for nome, conteudo in alteracoes.items():
                with self.subTest(alteracao=nome):
                    self.assertNotEqual(conteudo, self.lf)
                    self.assertNotEqual(sha256_canonico(conteudo), SHA256_BASE_CANONICA)
                    caminho = Path(pasta) / f'{nome}.csv'
                    caminho.write_bytes(conteudo)
                    with self.assertRaises(ValueError):
                        ler_validar(caminho)

if __name__ == '__main__':
    unittest.main()
