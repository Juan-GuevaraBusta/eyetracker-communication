"""Pruebas del rechazo de parpadeo (MVP) y del fallback de muestras."""

import unittest

import numpy as np

import calibration as calib
from gaze import detect_blink


class TestDetectBlink(unittest.TestCase):
    def test_un_solo_ojo_por_encima_del_umbral_no_cuenta_como_parpadeo(self):
        self.assertFalse(detect_blink(0.90, 0.10))
        self.assertFalse(detect_blink(0.10, 0.90))

    def test_ambos_ojos_cerrados_si_cuentan_como_parpadeo(self):
        self.assertTrue(detect_blink(0.90, 0.90))

    def test_scores_bajos_no_son_parpadeo(self):
        self.assertFalse(detect_blink(0.20, 0.20))

    def test_rechazo_desactivado_nunca_parpadea(self):
        self.assertFalse(detect_blink(0.99, 0.99, enabled=False))


class TestSampleFallback(unittest.TestCase):
    def test_fallback_usa_muestras_con_parpadeo_si_faltan_validas(self):
        validas = [np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0])]
        cualquieras = validas + [np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])]
        chosen = calib.choose_point_samples(validas, cualquieras, min_valid=8)
        self.assertEqual(len(chosen), 2)

    def test_fallback_prefiere_validas_si_alcanzan_el_minimo(self):
        validas = [np.ones(6) * i for i in range(8)]
        cualquieras = validas + [np.zeros(6)]
        chosen = calib.choose_point_samples(validas, cualquieras, min_valid=8)
        self.assertEqual(len(chosen), 8)

    def test_aggregate_con_pocas_muestras_no_descarta_el_punto(self):
        samples = [np.array([0.4, 0.5, 0.4, 0.5, 0.01, -0.02])]
        result = calib.aggregate_samples(samples)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, samples[0])


if __name__ == "__main__":
    unittest.main()
