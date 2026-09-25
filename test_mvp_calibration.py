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
        self.assertTrue(detect_blink(0.26, 0.26))

    def test_scores_bajos_no_son_parpadeo(self):
        self.assertFalse(detect_blink(0.20, 0.20))
        self.assertFalse(detect_blink(0.25, 0.25))

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


class TestBlinkFrame(unittest.TestCase):
    def test_marco_negro_solo_mientras_dura_el_parpadeo(self):
        from main import blink_frame_active

        self.assertFalse(blink_frame_active(now=1.0, last_blink_at=None))
        self.assertTrue(blink_frame_active(now=1.0, last_blink_at=1.0, hold_s=0.3))
        self.assertTrue(blink_frame_active(now=1.2, last_blink_at=1.0, hold_s=0.3))
        self.assertFalse(blink_frame_active(now=1.4, last_blink_at=1.0, hold_s=0.3))

    def test_dibuja_negro_alrededor_y_deja_el_centro(self):
        from main import draw_blink_frame

        canvas = np.full((200, 300, 3), 255, dtype=np.uint8)
        draw_blink_frame(canvas, margin=20)
        # Esquinas / borde: negro
        self.assertTrue(np.array_equal(canvas[5, 5], (0, 0, 0)))
        self.assertTrue(np.array_equal(canvas[5, 150], (0, 0, 0)))
        self.assertTrue(np.array_equal(canvas[100, 5], (0, 0, 0)))
        # Centro de la grilla: sigue blanco
        self.assertTrue(np.array_equal(canvas[100, 150], (255, 255, 255)))


class TestBlinkSelectsCell(unittest.TestCase):
    def test_sin_parpadeo_no_cambia_el_cuadro(self):
        from main import next_selected_cell

        self.assertEqual(next_selected_cell(selected=5, gazed=8, blink_now=False, blink_prev=False), 5)

    def test_al_parpadear_usa_casilla_con_ojos_abiertos_no_el_iris_cerrado(self):
        from main import gaze_cell_for_blink_mark

        # El iris del frame de parpadeo saltó a 8; la mirada estable era 5.
        self.assertEqual(gaze_cell_for_blink_mark(last_open=5, instant=8, eyes_closed=True), 5)
        self.assertEqual(gaze_cell_for_blink_mark(last_open=5, instant=8, eyes_closed=False), 8)
        self.assertIsNone(gaze_cell_for_blink_mark(last_open=None, instant=8, eyes_closed=True))

    def test_parpadeo_sobre_otro_cuadro_lo_cambia(self):
        from main import next_selected_cell

        self.assertEqual(next_selected_cell(selected=5, gazed=9, blink_now=True, blink_prev=False), 9)

    def test_mantener_ojos_cerrados_no_salta_de_cuadro(self):
        from main import next_selected_cell

        self.assertEqual(next_selected_cell(selected=5, gazed=9, blink_now=True, blink_prev=True), 5)

    def test_relleno_azul_solo_en_el_cuadro_elegido(self):
        from main import draw_selected_cell

        canvas = np.full((90, 90, 3), 255, dtype=np.uint8)
        draw_selected_cell(canvas, 1, 90, 90)
        # Centro del cuadro 1 (arriba-izquierda): azulado
        pixel = canvas[15, 15]
        self.assertGreater(int(pixel[0]), int(pixel[2]))
        # Centro del cuadro 9 (abajo-derecha): sigue blanco
        self.assertTrue(np.array_equal(canvas[75, 75], (255, 255, 255)))


class TestCellLabels(unittest.TestCase):
    def test_cada_categoria_tiene_nueve_opciones_y_volver(self):
        from main import CATEGORIES, SUBMENUS

        self.assertEqual(set(CATEGORIES), set(range(1, 10)))
        self.assertEqual(set(SUBMENUS), set(range(1, 10)))
        for cat, items in SUBMENUS.items():
            self.assertEqual(set(items), set(range(1, 10)), cat)
            self.assertIn("VOLVER", items[9]["label"])

    def test_frases_clinicas_caben_en_la_tarjeta(self):
        from main import SUBMENUS, label_layout_for_cell, CELL_LABEL_FONT
        import cv2

        cell_w, cell_h = 420, 140
        phrases = [
            item["label"]
            for items in SUBMENUS.values()
            for item in items.values()
        ]
        for text in phrases:
            lines, scale, thickness = label_layout_for_cell(text, cell_w, cell_h, max_scale=0.95)
            max_w = 0
            total_h = 0
            gap = int(max(4, scale * 10))
            for i, line in enumerate(lines):
                (tw, th), _ = cv2.getTextSize(line, CELL_LABEL_FONT, scale, thickness)
                max_w = max(max_w, tw)
                total_h += th
                if i:
                    total_h += gap
            self.assertLessEqual(max_w, int(cell_w * 0.92), text)
            self.assertLessEqual(total_h, int(cell_h * 0.85), text)
            self.assertGreaterEqual(scale, 0.35, text)


class TestMenuNavigation(unittest.TestCase):
    def test_parpadeo_en_el_menu_entra_a_la_categoria(self):
        from main import apply_menu_selection

        category, selected, phrase = apply_menu_selection(
            active_category=None, selected_cell=1, prev_selected=None, blink_edge=True,
        )
        self.assertEqual(category, 1)
        self.assertIsNone(selected)
        self.assertIsNone(phrase)

    def test_parpadeo_en_subopcion_confirma_la_frase(self):
        from main import apply_menu_selection

        category, selected, phrase = apply_menu_selection(
            active_category=1, selected_cell=1, prev_selected=None, blink_edge=True,
        )
        self.assertEqual(category, 1)
        self.assertEqual(selected, 1)
        self.assertEqual(phrase, "Tengo sed")

    def test_casilla_volver_regresa_al_menu(self):
        from main import apply_menu_selection

        category, selected, phrase = apply_menu_selection(
            active_category=9, selected_cell=9, prev_selected=None, blink_edge=True,
        )
        self.assertIsNone(category)
        self.assertIsNone(selected)
        self.assertIsNone(phrase)

    def test_sin_flanco_de_parpadeo_no_navega(self):
        from main import apply_menu_selection

        category, selected, phrase = apply_menu_selection(
            active_category=None, selected_cell=4, prev_selected=4, blink_edge=False,
        )
        self.assertIsNone(category)
        self.assertEqual(selected, 4)
        self.assertIsNone(phrase)


if __name__ == "__main__":
    unittest.main()
